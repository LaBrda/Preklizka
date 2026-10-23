FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY web ./web
COPY tools ./tools
COPY VERSION ./VERSION
RUN python tools/build_web.py

ARG APP_VERSION=dev
ENV APP_VERSION=${APP_VERSION}
ENV PREKLIZKA_MODELS=/data/modely
RUN mkdir -p /data/modely && chown nobody /data/modely
VOLUME /data/modely
EXPOSE 8080
WORKDIR /srv/app
USER nobody
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
