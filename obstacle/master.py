from globals import Master, Vernie, traceback, load_message, load_image, OBST_REQUEST, cv2


class ObstMaster(Master):
    def __init__(self):
        super(ObstMaster, self).__init__(OBST_REQUEST)
        self.mq.channel.basic_consume(queue=OBST_REQUEST, on_message_callback=self.process)
        self.mq.channel.start_consuming()

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
        image = self.receive(body)
        cv2.imshow('images', image)
        cv2.waitKey(0)


def main():
    master = ObstMaster()


if __name__ == "__main__":
    main()
