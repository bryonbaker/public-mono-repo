#! /bin/bash

podman run --rm -p 8080:8080   -e KC_BOOTSTRAP_ADMIN_USERNAME=admin   -e KC_BOOTSTRAP_ADMIN_PASSWORD=change_me   -v "$(pwd):/keycloak/rhbk-26.4.6/data/h2:Z"   quay.io/brbaker/keycloak:v0.1