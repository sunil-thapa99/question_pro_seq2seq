FROM ubuntu:22.04
MAINTAINER sunil43thapa@gmail.com

RUN apt-get update -y

RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.6


RUN apt-get install python3-pip -y
RUN apt-get install gunicorn3 -y


COPY requirements.txt requirements.txt
COPY . /opt/


RUN pip3 install -r requirements.txt
RUN python3 -m spacy download en_core_web_md

WORKDIR /opt/


CMD ["gunicorn3", "-b", "127.0.0.1:5005", "api:app"]