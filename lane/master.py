import tensorflow as tf

from globals import Master, Vernie, traceback, load_message, load_image, LANE_REQUEST


class LaneMaster(Master):
    def __init__(self, model):
        super(LaneMaster, self).__init__(LANE_REQUEST)
        self.mq.channel.basic_consume(queue=LANE_REQUEST, on_message_callback=self.process)
        self.mq.channel.start_consuming()
        self.model = model

    def receive(self, message):
        try:
            code, message, data = load_message(message)
            if code != 200:
                raise Vernie(300, 'Illegal message')
            return load_image(data)
        except Vernie:
            raise Vernie(300, 'Illegal message')
        except Exception:
            raise Vernie(301, 'Failed to receive message', traceback.format_exc())

    def process(self, ch, method, props, body):
        self.log.info(method.routing_key + ' received ' + props.correlation_id)
        data = self.receive(body)
        image = tf.convert_to_tensor(data, dtype=tf.float32)


class Lane(object):
    def __init__(self):
        self.model = tf.train.import_meta_graph('./model/model.meta')
        with tf.Session() as session:
            self.model.restore(session, tf.train.latest_checkpoint('./model/model'))


def main():
    master = LaneMaster(Lane())


if __name__ == "__main__":
    main()
