# Overview

This app provides a way to build a developer instance of Keycloak as a container that can run on OpenShift or as a podman container.

## Building

Before building:

1. Update the Makefile with the repo and label you want to use.

```bash
make build
```

2. Download the Keycloak release from Red Hat. The url can be found in the Keycloak documentation. The `rhbk-<release>.zip` file must exist in the same directory as the `Containerfile`. This build uses release `26.4.6`.

## Running locally

Run this from the directory where you want the Keycloak database to be saved. If not, update the volume path accordingly.

```bash
podman run --rm -p 8080:8080 \
   --name keycloak \
   -e KC_BOOTSTRAP_ADMIN_USERNAME=admin \
   -e KC_BOOTSTRAP_ADMIN_PASSWORD=change_me \
   -v "$(pwd):/keycloak/rhbk-26.4.6/data/h2:Z" \
   quay.io/brbaker/keycloak:latest
```

## Debugging

If the app does not come up then the logs will be found via:

```bash
journalctl -f
```

```bash
podman logs keycloak
```
