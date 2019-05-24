from globals import Master, load_image, MQ, NN, LOG
from obstacle import *


class ObstMaster(Master):
    def __init__(self):
        super(ObstMaster, self).__init__(MQ.OBST_REQUEST)
        with tf.device("/gpu:0"):
            self.model = modellib.MaskRCNN(mode="inference", model_dir=LOG.DIR, config=coco.CocoConfig())
        self.model.load_weights(NN.OBST_WEIGHTS_PATH, by_name=True)
        self.queue = MQ.OBST_REQUEST
        self.mq.channel.basic_consume(self.queue, on_message_callback=self.process)
        self.mq.channel.start_consuming()

    def process(self, ch, method, props, body):
        try:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            self.log.info(method.routing_key + ' received message')
            image = load_image(body)
            result = self.model.detect([image], verbose=1)[0]
            image = draw_result(image, result)
            data = cv2.imencode('.jpg', image)[1].tostring()
            self.mq.publish(MQ.OBST_RESPONSE, data)
        except Exception as err:
            self.log.error(err)


def main():
    master = ObstMaster()


if __name__ == "__main__":
    main()
