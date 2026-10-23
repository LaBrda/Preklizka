# Překližkové vrstvy

Z obrázku postavičky udělá řezné SVG pro laser se 3 až 5 vrstvami (černá deska, natřené vrstvy,
horní vrstva z čistého dřeva s rámečkem). Ovládání běží v prohlížeči, přesný výpočet vrstev dělá
server ve vyšším rozlišení (hladké řezy). Bez serveru se počítá přímo v prohlížeči.

## Windows: Preklizka.exe (bez instalace)

1. Na stránce **Releases** repozitáře stáhni `Preklizka.exe` z nejnovější verze.
2. Spusť ho. Otevře se okno s informací a appka v prohlížeči (`http://127.0.0.1:8765`).
   Windows může poprvé ukázat „Systém Windows ochránil počítač“ → *Další informace* → *Přesto spustit*.
3. Aktualizace: v appce tlačítko **Aktualizovat**. Exe stáhne novou verzi, vymění se a restartuje.
4. Ukončení: zavři černé okno.

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
Strop je nastavený v `docker-compose.yml` (2 jádra, 1,5 GB). Najednou se počítá jen jeden obrázek.

## Vývoj

- `web/page.html` + `web/core.js` – ovládání a výpočet v prohlížeči (`tools/build_web.py` z nich složí `web/index.html`)
- `app/vrstvy.py` – výpočet vrstev na serveru, `app/server.py` – FastAPI
- lokálně: `pip install -r requirements.txt && python tools/build_web.py && cd app && uvicorn server:app --port 8080`
