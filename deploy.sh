#!/bin/bash
# Deploy the checked-out main to this VM as a Docker container (app only; Postgres is Supabase).
# Run by .github/workflows/deploy.yml on every push to main, or by hand.
# Requires: docker + compose plugin, key/airtc-prod.json, and a .env with DATABASE_URL and API keys.
set -euo pipefail

cd /home/ubuntu/AIRTC
COMPOSE="docker compose -f docker-compose.yml -f docker-compose.prod.yml"
HEALTH_TIMEOUT=90   # seconds to wait for the new container to report healthy

git pull origin main

# Keep the image that is serving right now so a bad build can be rolled back in seconds.
if docker image inspect airtc:prod >/dev/null 2>&1; then
    docker tag airtc:prod airtc:prev
fi

# Rebuild the app image from source (cached layers make this fast when only code changed),
# start/refresh the container. restart: unless-stopped keeps it alive across reboots.
$COMPOSE up -d --build

# Gate on the Dockerfile HEALTHCHECK (HTTP GET / inside the container) rather than on
# "the container started": a crash loop or an app that never binds 8081 must fail the deploy.
container=$($COMPOSE ps -q app)
deadline=$((SECONDS + HEALTH_TIMEOUT))
status="starting"
while [ "$SECONDS" -lt "$deadline" ]; do
    status=$(docker inspect --format '{{.State.Health.Status}}' "$container" 2>/dev/null || echo "missing")
    [ "$status" = "healthy" ] && break
    sleep 5
done

if [ "$status" != "healthy" ]; then
    echo "DEPLOY FAILED: container is '$status' after ${HEALTH_TIMEOUT}s. Last log lines:" >&2
    $COMPOSE logs --no-log-prefix --tail 40 app >&2 || true
    if docker image inspect airtc:prev >/dev/null 2>&1; then
        echo "Rolling back to the previous image." >&2
        docker tag airtc:prev airtc:prod
        $COMPOSE up -d --no-build app
    fi
    exit 1
fi

# Drop images left behind by previous builds so they don't accumulate on disk.
# airtc:prev is tagged, so it survives the prune and stays available for a manual rollback.
docker image prune -f

$COMPOSE ps
echo "Deployed $(git rev-parse --short HEAD); container healthy."
