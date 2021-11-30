FROM nvidia/cuda:11.1.1-cudnn8-devel
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        git \
        vim \
        curl \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        rsync \
        software-properties-common \
        sudo \
        sqlite3 \
        zip \
        unzip \
        rar \
        unrar \
        apache2-utils \
        nano
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip3 --no-cache-dir install --upgrade pip
COPY requirements.txt .
RUN pip3 --no-cache-dir install -r requirements.txt && \
	rm requirements.txt
RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git'
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN mkdir /workspace
WORKDIR /workspace
# ENTRYPOINT bash
CMD ["python3", "/workspace/bin/run.py"]
