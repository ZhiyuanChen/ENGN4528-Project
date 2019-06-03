FROM pytorch/1.1.0-cuda10.0-cudnn7.5-devel


RUN conda install -y -c \
  pygpu \
  pyyaml \
  opencv3 \
  numpy \
  && \

ADD theanorc /home/keras/.theanorc

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PYTHONPATH='/src/:$PYTHONPATH'


ENV EMARKING=/usr/local/emarking
RUN mkdir $EMARKING
COPY ./build/script $EMARKING/script
COPY ./models /opt/models
COPY ./build/lib/engine_python.so  /usr/local/lib/python2.7/dist-packages/engine_python.so
COPY ./build/tesseract/chi_sim.traineddata /usr/share/tesseract-ocr/4.00/tessdata/


VOLUME ["/home/vol"]
WORKDIR $EMARKING/script

ENTRYPOINT ["bash", "-c", "python $EMARKING/script/TestPaperManager.py"]
