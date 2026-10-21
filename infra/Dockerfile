# syntax=docker/dockerfile:1.7
# FieldOps API / worker image.
#
#   docker build -f infra/Dockerfile -t fieldops:dev .
#   docker run --rm -p 8000:8000 --env-file .env fieldops:dev
#
# Same image for the API (default CMD) and the Celery worker (override the command).
# Pin base images by digest in production; tags here keep the course readable.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first (cached unless the lockfile changes), then the project.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra observability
COPY capstone ./capstone
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra observability

# The demo bakes in the synthetic world. A real deployment reads tenants/customers from
# Postgres and the CRM API instead.
RUN .venv/bin/python -m capstone.data.synthetic.generator --out capstone/data/synthetic/out


FROM python:3.12-slim-bookworm AS runtime
RUN groupadd --system app && useradd --system --gid app --uid 10001 --no-create-home app
WORKDIR /app
COPY --from=build --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"
# --proxy-headers: Container Apps terminates TLS at the ingress.
CMD ["uvicorn", "capstone.app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--proxy-headers", "--forwarded-allow-ips", "*"]
