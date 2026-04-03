FROM postgres:18-bookworm

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        postgresql-18-pgvector \
    && rm -rf /var/lib/apt/lists/*
