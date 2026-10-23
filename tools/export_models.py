"""Převod AI modelů do ONNX (běží v GitHub Actions, workflow models.yml).

Výstup ve složce models_out/ se nahraje do GitHub Release "modely"; appka si modely stahuje odtud.
"""
import os
import sys
import urllib.request

import torch
from torch import nn
import torch.nn.functional as F

OUT = "models_out"
SRC = "models_src"
os.makedirs(OUT, exist_ok=True)
os.makedirs(SRC, exist_ok=True)

URLS = {
    "RealESRGAN_x4plus_anime_6B.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "realesr-animevideov3.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
    "face_paint_512_v2.pt": "https://raw.githubusercontent.com/bryandlee/animegan2-pytorch/main/weights/face_paint_512_v2.pt",
    "paprika.pt": "https://raw.githubusercontent.com/bryandlee/animegan2-pytorch/main/weights/paprika.pt",
    "animegan_model.py": "https://raw.githubusercontent.com/bryandlee/animegan2-pytorch/main/model.py",
}
# hotové ONNX modely jen zkopírovat (zrcadlo, ať appka stahuje z jednoho místa)
MIRROR = {
    "isnet-anime.onnx": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-anime.onnx",
    "isnet-general-use.onnx": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
    "sam-encoder.onnx": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/vit_b-encoder-quant.onnx",
    "sam-decoder.onnx": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/vit_b-decoder-quant.onnx",
}


def get(name, url, folder):
    p = os.path.join(folder, name)
    if not os.path.exists(p):
        print("stahuji", url)
        urllib.request.urlretrieve(url, p)
    return p


# ---- Real-ESRGAN (architektury podle basicsr) ----
class RDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = RDB(nf, gc), RDB(nf, gc), RDB(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, nb=6, nf=64, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2.0, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2.0, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class SRVGGNetCompact(nn.Module):
    def __init__(self, nf=64, nc=16, up=4):
        super().__init__()
        self.up = up
        body = [nn.Conv2d(3, nf, 3, 1, 1), nn.PReLU(num_parameters=nf)]
        for _ in range(nc):
            body += [nn.Conv2d(nf, nf, 3, 1, 1), nn.PReLU(num_parameters=nf)]
        body += [nn.Conv2d(nf, 3 * up * up, 3, 1, 1)]
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(up)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=float(self.up), mode="nearest")


def export(model, name, size=64):
    model.eval()
    x = torch.rand(1, 3, size, size)
    p = os.path.join(OUT, name)
    torch.onnx.export(model, x, p, input_names=["input"], output_names=["output"], opset_version=17,
                      dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "H", 3: "W"}})
    print("hotovo", p, os.path.getsize(p))


def main():
    for n, u in URLS.items():
        get(n, u, SRC)
    for n, u in MIRROR.items():
        get(n, u, OUT)

    m = RRDBNet(nb=6)
    sd = torch.load(os.path.join(SRC, "RealESRGAN_x4plus_anime_6B.pth"), map_location="cpu")
    m.load_state_dict(sd.get("params_ema", sd.get("params", sd)))
    export(m, "esrgan-anime.onnx")

    m = SRVGGNetCompact()
    sd = torch.load(os.path.join(SRC, "realesr-animevideov3.pth"), map_location="cpu")
    m.load_state_dict(sd.get("params", sd.get("params_ema", sd)))
    export(m, "esrgan-fast.onnx")

    sys.path.insert(0, SRC)
    import animegan_model
    for pt, name in [("face_paint_512_v2.pt", "cartoon-portret.onnx"), ("paprika.pt", "cartoon-krajina.onnx")]:
        g = animegan_model.Generator()
        g.load_state_dict(torch.load(os.path.join(SRC, pt), map_location="cpu"))
        export(g, name, 256)


if __name__ == "__main__":
    main()
