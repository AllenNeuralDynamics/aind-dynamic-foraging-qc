# hash:sha256:73f177753719fc953430ccfc46c3b36a8a9dd99f83da4d0be9e7826d9638d1d5
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-access-api[docdb]==0.17.1 \
    aind-data-schema==1.3.0 \
    aind-data-schema-models==0.7.5 \
    aind-log-utils==0.2.4 \
    aws-requests-auth==0.4.3 \
    boto3==1.36.2 \
    kachery-cloud==0.4.10 \
    matplotlib==3.7.5 \
    numpy==1.24.4 \
    pandas==2.0.3
