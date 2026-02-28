# .devcontainer/Dockerfile
ARG BASE_IMAGE=pytorch/pytorch:2.3.1-cuda12.1-cudnn9-devel
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    curl \
    ssh \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace




