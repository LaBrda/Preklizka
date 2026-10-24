# Překližkové vrstvy

Z obrázku postavičky udělá řezné SVG pro laser se 3 až 5 vrstvami (černá deska, natřené vrstvy,
horní vrstva z čistého dřeva s rámečkem). Ovládání běží v prohlížeči, přesný výpočet vrstev dělá
server ve vyšším rozlišení (hladké řezy). Bez serveru se počítá přímo v prohlížeči.

## Windows: Preklizka.exe (bez instalace)

1. Na stránce **Releases** repozitáře stáhni `Preklizka.exe` z nejnovější verze.
2. Spusť ho. Appka se otevře ve vlastním okně (Edge WebView2, ve Windows 10/11 už je).
   Windows může poprvé ukázat „Systém Windows ochránil počítač“ → *Další informace* → *Přesto spustit*.
   Kdyby okno nešlo otevřít, appka se otevře v prohlížeči (vynutit: `Preklizka.exe --prohlizec`).
3. Aktualizace: v appce tlačítko **Aktualizovat**. Exe stáhne novou verzi, vymění se a restartuje.
4. Ukončení: zavři okno.

## 3D model (jen Preklizka.exe)

Přepínač **Obrázek / 3D model** v hlavičce. 3D režim nařeže autíčko apod. na vrstvy, které se lepí vedle sebe
(kolíky, vydlabání, čísla vrstev, gravír detailů z boční fotky na viditelné boky), a uloží SVG desky pro laser.
Vstup: fotky bok + předek (+ shora) na jednobarevném pozadí, nebo 3D sken OBJ/STL/GLB/PLY
(např. z RealityScan: 50–100 fotek dokola, výpočet na grafice). Na serveru v Dockeru se režim nezobrazuje.

## AI pomocník

- **Pozadí pryč** (kreslený obrázek / fotka), **Vybrat kliknutím** (ohraničit postavu tažením, klik přidá, Shift+klik ubere),
  **Zvětšit a vyčistit** (4×, pro malé a zubaté obrázky), **Fotka → kreslený obrázek** (styl portrét / scéna / jen zjednodušit).
- Modely (ONNX, release `modely`) se stáhnou při prvním použití: u exe do složky `Preklizka_modely` vedle exe, v Dockeru do svazku `modely`.
- Exe počítá na grafické kartě přes DirectML (NVIDIA, AMD, Intel), jinak na procesoru. Funguje i na notebooku bez RTX, jen pomaleji.
- Modely: Real-ESRGAN (BSD-3), AnimeGANv2 (bryandlee/animegan2-pytorch), IS-Net a Segment Anything přes rembg. Převod: `tools/export_models.py`.

## Instalace na Proxmox (Debian LXC s Dockerem)

1. **Kontejner.** V Proxmoxu vytvoř Debian 12 LXC: 2 jádra, 1,5 GB RAM, 6 GB disk.
   V *Options → Features* zapni **nesting** a **keyctl** (Docker to v LXC potřebuje).
2. **Docker.** V konzoli kontejneru:
   ```sh
   apt update && apt install -y curl git
   curl -fsSL https://get.docker.com | sh
   ```
3. **Přístup k obrazu.** Obraz `ghcr.io/labrda/preklizka` je veřejný, nic se nepřihlašuje.
   (Kdyby byl soukromý: token s oprávněním `read:packages` a `docker login ghcr.io -u LaBrda`.)
4. **Appka.**
   ```sh
   mkdir -p /opt/vrstvy && cd /opt/vrstvy
   curl -fsSLO https://raw.githubusercontent.com/LaBrda/preklizka/main/docker-compose.yml   # nebo soubor zkopíruj ručně
   echo "WATCHTOWER_TOKEN=$(openssl rand -hex 24)" > .env
   echo "UPDATE_PASSWORD=" >> .env      # volitelně heslo pro tlačítko Aktualizovat
   docker compose up -d
   ```
   (Je-li repozitář soukromý, `docker-compose.yml` zkopíruj z GitHubu ručně, `curl` bez tokenu nedosáhne.)
5. Otevři `http://<IP kontejneru>:8080`. Vpravo nahoře uvidíš verzi serveru a tlačítko **Aktualizovat**.

## Aktualizace

- Každá změna v repozitáři (větev `main`) se přes GitHub Actions sestaví do obrazu
  `ghcr.io/labrda/preklizka:latest`.
- V appce klikni na **Aktualizovat**: Watchtower stáhne novou verzi a appku restartuje.
- Ručně: `cd /opt/vrstvy && docker compose pull && docker compose up -d`.

## Zátěž serveru

V klidu ~60 MB RAM a téměř nulový procesor. Při výpočtu 1–2 jádra na několik sekund a až ~1 GB RAM.
Strop je nastavený v `docker-compose.yml` (2 jádra, 2,5 GB kvůli AI). Najednou se počítá jen jeden obrázek.
AI funkce na serveru běží na procesoru (pozadí ~5 s, zvětšení ~5–20 s), každá v samostatném procesu, takže paměť se po ní vrátí.

## Vývoj

- `web/page.html` + `web/core.js` – ovládání a výpočet v prohlížeči (`tools/build_web.py` z nich složí `web/index.html`)
- `app/vrstvy.py` – výpočet vrstev na serveru, `app/server.py` – FastAPI
- lokálně: `pip install -r requirements.txt && python tools/build_web.py && cd app && uvicorn server:app --port 8080`
