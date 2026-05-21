FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel as builder

ARG UID
ARG GID
ARG USERNAME=dsvolkov

RUN apt-get update \
    && apt-get install -y --no-install-recommends apt-utils \
    && apt-get install libgomp1 build-essential pandoc -y \
    && apt-get install git -y --no-install-recommends \
    && apt-get install -y \
    bash \
    bash-completion \
    git \
    wget \
    curl \
    vim \
    unzip \
    sudo \
    build-essential \
    tmux\
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g "${GID}" "${USERNAME}" \
  && useradd --create-home --no-log-init -u "${UID}" -g "${GID}" -s /bin/bash "${USERNAME}"

WORKDIR /workspace

RUN pip install --no-cache-dir --upgrade pip wheel

COPY requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt

USER "${USERNAME}"

CMD ["bash"]