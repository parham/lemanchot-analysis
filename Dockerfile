FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Montreal
RUN apt-get update && apt-get install -y apt-utils apt-transport-https git wget zip build-essential cmake vim
RUN apt-get remove python-* && apt-get autoremove
RUN apt-get install -y python3 python3-dev python3-pip

RUN touch /root/.bashrc && echo 'alias python=python3' >> /root/.bashrc

RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

RUN --mount=type=ssh git clone git@github.com:parham/lemanchot-analysis.git
RUN cd lemanchot-analysis && pip install -r requirements.txt
WORKDIR /lemanchot-analysis

 CMD ["git", "pull", "&&", "tail", "-f", "/dev/null"]
