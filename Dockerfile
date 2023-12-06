FROM python:3.9.6
MAINTAINER sunil43thapa@gmail.com

ENV DOCKER_DEFAULT_PLATFORM=linux/amd64

WORKDIR /opt/

COPY api.py /opt/
COPY api_helper.py /opt/
COPY vocabs /opt/vocabs
COPY models /opt/models
COPY requirements.txt requirements.txt

RUN apt-get update -y &&\
    python -m venv ./venv &&\
    chmod -R 755 . &&\
    apt-get install python3-pip -y && \
    ./venv/bin/activate &&\
    ./venv/bin/pip install -r requirements.txt &&\
    ./venv/bin/python -m spacy download en_core_web_md

CMD ./venv/bin/gunicorn -b 0.0.0.0:8000 api:app --timeout 120