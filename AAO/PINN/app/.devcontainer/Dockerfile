# Base docker image
FROM ubuntu:22.04

### Define default arguments
# Are overwritten by build arguments (if supplied while building the container)
ARG uid=1000
ARG gid=1000
ARG requirements=requirements.txt
ARG requirements_dev=requirements-development.txt


### Change default shell
SHELL ["/bin/bash", "--login", "-c"]


### Create a non-root user
# For more information, see:
# - https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
# - https://vsupalov.com/docker-shared-permissions/
ENV USER_NAME docker_user
ENV USER_UID $uid
ENV USER_GID $gid
ENV HOME_DIR /home/$USER_NAME

RUN groupadd --gid $USER_GID $USER_NAME

RUN adduser \
    --disabled-password \
    --gecos "non-root user" \
    --uid $USER_UID \
    --gid $USER_GID \
    --home $HOME_DIR \
    $USER_NAME


### Copy files
# Requirements
COPY $requirements /tmp/
# Development requirements
COPY $requirements_dev /tmp/


### Install base utilities
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y g++ && \
    apt-get install -y wget && \
    apt-get install -y nano && \
    apt-get install -y python3-dev && \
    apt-get install -y git

### Install dependencies
RUN mkdir /build

# Install Python
ENV PYTHON_RELEASE 3.11.1
ENV PYTHON_VERSION 3.11
RUN cd /build && \
    apt-get install -y zlib1g-dev libgdbm-dev libncurses5-dev libssl-dev libnss3-dev libffi-dev libreadline-dev libsqlite3-dev libbz2-dev && \
    wget https://www.python.org/ftp/python/${PYTHON_RELEASE}/Python-${PYTHON_RELEASE}.tgz && \
    tar -xf Python-${PYTHON_RELEASE}.tgz && \
    cd Python-${PYTHON_RELEASE} && \
    ./configure --enable-optimizations --with-ensurepip=install && \
    make -j $(nproc) && \
    make altinstall && \
    ln -sf /usr/local/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -sf /usr/local/bin/pip${PYTHON_VERSION} /usr/bin/pip

#Clean up
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*


### Install Python requirements
RUN pip install --upgrade pip
RUN pip install -r /tmp/$requirements
RUN pip install -r /tmp/$requirements_dev


# Install jax
RUN pip install --upgrade jax


### Create working directory
ENV APP_DIR $HOME_DIR/app
RUN mkdir -p $APP_DIR


### Change to non-root user and set working directory
USER $USER_NAME
WORKDIR $APP_DIR
