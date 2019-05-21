import tensorflow as tf

from obstacle import *
from globals import Master, Vernie, traceback, load_image, MQ, NN, LOG, cv2


class ObstMaster(Master):
    def __init__(self):
        super(ObstMaster, self).__init__(MQ.OBST_REQUEST)
        with tf.device("/gpu:0"):
            self.model = modellib.MaskRCNN(mode="inference", model_dir=LOG.DIR, config=coco.CocoConfig())
        self.model.load_weights(NN.OBST_WEIGHTS_PATH, by_name=True)
        self.mq.channel.basic_consume(queue=MQ.OBST_REQUEST, on_message_callback=self.process)
        self.mq.channel.start_consuming()

    def process(self, ch, method, props, body):
        self.log.info(method.routing_key + ' received ' + props.correlation_id)
        image = load_image(body)
        result = self.model.detect([image], verbose=1)[0]


class Obstacle(object):
    def __init__(self):
        pass


def main():
    master = ObstMaster()


if __name__ == "__main__":
    main()
