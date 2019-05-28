FROM pytorch/1.1.0-cuda10.0-cudnn7.5-devel


RUN pip install --upgrade pip && \
    pip install \
      pika \
      tensorflow-gpu \
      && \
    conda install -y -c \
        pygpu \
        pyyaml \
        opencv3 \
        numpy \
      && \

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PYTHONPATH='/src/:$PYTHONPATH'


ENV LANE=/usr/local/lane
RUN mkdir $LANE
COPY ./global $LANE/global
COPY ./objects /$LANE/objects
COPY ./lane /$LANE/lane

VOLUME ["/home/vol"]
WORKDIR $LANE/

ENTRYPOINT ["bash", "-c", "python $LANE/lane/master.py"]
