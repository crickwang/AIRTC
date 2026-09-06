#!/bin/bash
# Deploy the checked-out main to this VM as a Docker container (app only; Postgres is Supabase).
# Requires: docker + compose plugin, key/airtc-prod.json, and a .env with DATABASE_URL and API keys.
set -euo pipefail

cd /home/ubuntu/AIRTC
git pull origin main

# Rebuild the app image from source (cached layers make this fast when only code changed),
# start/refresh the container. restart: unless-stopped keeps it alive across reboots.
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

# Drop images left behind by previous builds so they don't accumulate on disk.
docker image prune -f

docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
