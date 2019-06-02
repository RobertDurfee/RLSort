FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python-opengl \
    xvfb

RUN pip3 install \
    numpy \
    matplotlib \
    pandas \
    scikit-learn \
    jupyter \
    gym

WORKDIR /workspace

COPY ./entrypoint.sh /
RUN chmod +x /entrypoint.sh

ENTRYPOINT [ "/bin/bash", "/entrypoint.sh" ]
