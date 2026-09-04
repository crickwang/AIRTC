#!/bin/bash
# Deploy the latest main to this VM as a Docker container.
# Requires: docker + compose plugin, and a .env with DATABASE_URL (Supabase) and API keys.
set -euo pipefail

cd /home/ubuntu/AIRTC
git pull origin main

# Rebuild from source (cached layers make this fast when only code changed),
# then replace the running container. restart: unless-stopped keeps it alive across reboots.
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build app

# Drop images left behind by previous builds so they don't accumulate on the 58 GB disk.
docker image prune -f

docker compose -f docker-compose.yml -f docker-compose.prod.yml ps app
