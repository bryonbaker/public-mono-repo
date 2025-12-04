#!/usr/bin/env bash
set -euo pipefail

cd /keycloak/rhbk-26.4.6

# Start Keycloak in dev mode in the background
./bin/kc.sh start-dev &
KC_PID=$!

# Wait for Keycloak to become ready
echo "Waiting for Keycloak to become ready on http://localhost:8080 ..."
until curl -sf http://localhost:8080/realms/master/.well-known/openid-configuration >/dev/null 2>&1; do
  sleep 2
done
echo "Keycloak is up, configuring realm SSL settings ..."

# If bootstrap admin env vars are provided, use them to log in and disable SSL
if [[ -n "${KC_BOOTSTRAP_ADMIN_USERNAME:-}" && -n "${KC_BOOTSTRAP_ADMIN_PASSWORD:-}" ]]; then
  /keycloak/rhbk-26.4.6/bin/kcadm.sh config credentials \
    --server http://localhost:8080 \
    --realm master \
    --user "${KC_BOOTSTRAP_ADMIN_USERNAME}" \
    --password "${KC_BOOTSTRAP_ADMIN_PASSWORD}"

  /keycloak/rhbk-26.4.6/bin/kcadm.sh update realms/master -s sslRequired=NONE || \
    echo "Warning: failed to set sslRequired=NONE on master realm"
else
  echo "Warning: KC_BOOTSTRAP_ADMIN_USERNAME and KC_BOOTSTRAP_ADMIN_PASSWORD not set; skipping sslRequired=NONE update"
fi

# Wait for Keycloak process so the container stays in the foreground
wait "${KC_PID}"

