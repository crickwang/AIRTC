#!/bin/bash
# Deploy the checked-out main to this VM as Docker containers (app + postgres).
# Requires: docker + compose plugin, and a .env with POSTGRES_PASSWORD, DATABASE_URL, API keys.
set -euo pipefail

cd /home/ubuntu/AIRTC
git pull origin main

# Rebuild the app image from source (cached layers make this fast when only code changed),
# start/refresh both containers. restart: unless-stopped keeps them alive across reboots.
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

# Drop images left behind by previous builds so they don't accumulate on disk.
docker image prune -f

docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
