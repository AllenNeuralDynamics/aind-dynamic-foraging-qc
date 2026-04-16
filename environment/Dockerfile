# hash:sha256:14a608ee47186ea807153add57ee424619a646a1b1be2dda799b4116aa0bf0a4
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

ENV PROCESS_NAME=aind-dynamic-foraging-qc

RUN pip install -U --no-cache-dir \
    aind-data-access-api[docdb]==0.17.1 \
    aind-data-schema==1.3.0 \
    aind-data-schema-models==0.7.5 \
    aws-requests-auth==0.4.3 \
    boto3==1.36.2 \
    kachery-cloud==0.4.10 \
    matplotlib==3.7.5 \
    numpy==1.24.4 \
    pandas==2.0.3 \
    pyyaml==6.0.3 \
    watchtower==3.4.0

ADD "https://github.com/coder/code-server/releases/download/v4.95.3/code-server-4.95.3-linux-amd64.tar.gz" /.code-server/code-server.tar.gz

RUN cd /.code-server \
    && tar -xvf code-server.tar.gz \
    && rm code-server.tar.gz \
    && ln -s /.code-server/code-server-4.95.3-linux-amd64/bin/code-server /usr/bin/code-server \
    && mkdir -p /.vscode/extensions
