import tensorflow as tf

from globals import Master, Vernie, traceback, load_image, MQ, NN


class LaneMaster(Master):
    def __init__(self):
        super(LaneMaster, self).__init__(MQ.LANE_REQUEST)
        self.mq.channel.basic_consume(queue=MQ.LANE_REQUEST, on_message_callback=self.process)
        self.mq.channel.start_consuming()

    def process(self, ch, method, props, body):
        self.log.info(method.routing_key + ' received ' + props.correlation_id)
        image = load_image(body) - NN.VGG_MEAN


class Lane(object):
    def __init__(self):
        pass


def main():
    master = LaneMaster(Lane())


if __name__ == "__main__":
    main()
