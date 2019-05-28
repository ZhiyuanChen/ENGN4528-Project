FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# Install conda
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN wget --quiet --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh && \
    echo "c59b3dd3cad550ac7596e0d599b91e75d88826db132e4146030ef471bb434e9a *Miniconda3-4.6.14-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash /Miniconda3-4.6.14-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-4.6.14-Linux-x86_64.sh && \

# Install Python packages and keras
ENV NB_USER ENGN4528
ENV NB_UID 1000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    chown $NB_USER $CONDA_DIR -R && \
    mkdir -p /src && \
    chown $NB_USER /src

USER $NB_USER

ARG python_version=3.6

RUN conda config --append channels conda-forge
RUN conda install -y python=${python_version} && \
    pip install --upgrade pip && \
    pip install \
      pika \
      tensorflow-gpu \
      imgaug \
      pycocotools \
      && \
    conda install \
      pygpu \
      pyyaml \
      opencv3 \
      numpy \
      && \
    git clone git://github.com/keras-team/keras.git /src && pip install -e /src[tests] && \
    pip install git+git://github.com/keras-team/keras.git && \
    conda clean -yt

ADD theanorc /home/keras/.theanorc

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PYTHONPATH='/src/:$PYTHONPATH'


ENV OBSTACLE=/usr/local/obstacle
RUN mkdir $OBSTACLE
COPY ./global $OBSTACLE/global
COPY ./objects /$OBSTACLE/objects
COPY ./obstacle /$OBSTACLE/obstacle

VOLUME ["/home/vol"]
WORKDIR $OBSTACLE/

ENTRYPOINT ["bash", "-c", "python $OBSTACLE/obstacle/master.py"]
