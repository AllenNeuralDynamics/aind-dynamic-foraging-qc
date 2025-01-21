# hash:sha256:01caf3e685c85d8d785ba0d3b60cfbf163cac15b04f8ad29c58cb204d0744231
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-access-api[docdb] \
    aind-data-schema \
    aind-data-schema-models \
    aind-log-utils \
    aws-requests-auth \
    boto3 \
    kachery-cloud \
    matplotlib \
    numpy \
    pandas
