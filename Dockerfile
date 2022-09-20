
FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Montreal

RUN apt-get update && apt-get install -y apt-utils apt-transport-https git wget zip build-essential cmake vim screen
RUN apt-get remove python-* && apt-get autoremove
RUN apt-get install -y python3 python3-dev python3-pip python-is-python3 

ADD . /lemanchot-analysis

RUN cd lemanchot-analysis && pip install -r requirements.txt
WORKDIR /lemanchot-analysis

RUN pip install git+https://github.com/qubvel/segmentation_models.pytorch \
    git+https://github.com/waspinator/coco.git@2.1.0 \
    git+https://github.com/waspinator/pycococreator.git@0.2.0 && pip install gimp-labeling-converter

CMD ["tail", "-f", "/dev/null"]
