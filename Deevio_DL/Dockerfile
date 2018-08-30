FROM ubuntu:16.04
LABEL maintainer caffe-maint@googlegroups.com

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN mkdir -p /workspace/src

WORKDIR /workspace/src
COPY ./requirements.txt /workspace/src/requirements.txt
RUN pip install -r requirements.txt

COPY . /workspace/src

VOLUME /workspace/data /workspace/model /workspace/temp /workspace/processed

RUN ulimit -c unlimited

CMD bash
EXPOSE 5000
ENTRYPOINT ["python"]
CMD ["app.py"]
