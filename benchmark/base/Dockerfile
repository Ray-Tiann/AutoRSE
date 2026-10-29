ARG NODE_BASE_IMAGE=node:22-bullseye-slim
FROM ${NODE_BASE_IMAGE} AS node-runtime
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CODEX_VERSION=latest
ARG CLAUDE_CODE_VERSION=latest
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates curl git jq procps python3 python3-pip ripgrep tmux \
    && rm -rf /var/lib/apt/lists/* \
    && tmux -V
COPY --from=node-runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=node-runtime /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -sf ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -sf ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && npm install --global "@openai/codex@${CODEX_VERSION}" \
       "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}"
RUN python3 -m pip install 'numpy==1.26.4' \
    && python3 -m pip install 'torch==2.1.2' --index-url https://download.pytorch.org/whl/cu121
COPY requirements.txt /tmp/autorse-requirements.txt
RUN python3 -m pip install -r /tmp/autorse-requirements.txt \
    && python3 -m pip check \
    && python3 -c 'import torch; assert torch.__version__.split("+")[0] == "2.1.2"'
WORKDIR /workspace
CMD ["/bin/bash"]
