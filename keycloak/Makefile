# Makefile for building standalone (dev) keycloak container

IMAGE_NAME := quay.io/brbaker/keycloak
TAG := v0.1.1

.PHONY: all build push clean

all: build push

build:
	podman build -t $(IMAGE_NAME):$(TAG) .
	podman tag $(IMAGE_NAME):$(TAG) $(IMAGE_NAME):latest

push:
	podman push $(IMAGE_NAME):$(TAG)
	podman push $(IMAGE_NAME):latest
clean:
	podman rmi -f $(IMAGE_NAME):$(TAG) || true
	podman rmi -f $(IMAGE_NAME):latest || true