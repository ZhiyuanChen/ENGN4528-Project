from globals import Master, Vernie, traceback, load_image, MQ, cv2


class SignMaster(Master):
    def __init__(self):
        super(SignMaster, self).__init__(MQ.SIGN_REQUEST)
        self.mq.channel.basic_consume(queue=MQ.SIGN_REQUEST, on_message_callback=self.process)
        self.mq.channel.start_consuming()

    def process(self, ch, method, props, body):
        self.log.info(method.routing_key + ' received message')
        image = load_image(body)
        cv2.imshow('image', image)
        cv2.waitKey()


def main():
    master = SignMaster()


if __name__ == "__main__":
    main()
