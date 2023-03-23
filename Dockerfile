FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu18.04

# LABEL org.opencontainers.image.source https://github.com/serengil/deepface
WORKDIR /app/

COPY . /app/

RUN apt-get update && \
    apt-get install -y build-essential git wget unzip software-properties-common && \
    apt-get install -y vim openssh-server libopenmpi-dev && \
    apt-get install -y python3.8 python3.8-distutils python3.8-dev && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache

RUN wget https://bootstrap.pypa.io/get-pip.py --progress=bar:force:noscroll --no-check-certificate \
    && python get-pip.py \
    && rm get-pip.py

RUN pip install .
RUN pip uninstall tensorflow -y
RUN pip install tensorflow-gpu

# CMD ["/bin/bash"]
CMD uvicorn --host=0.0.0.0 --port 8800 main:app