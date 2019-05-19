from globals import Master, Vernie, traceback, load_message, load_image, SIGN_REQUEST


class SignMaster(Master):
    def __init__(self):
        super(SignMaster, self).__init__(SIGN_REQUEST)
        self.mq.channel.basic_consume(queue=SIGN_REQUEST, on_message_callback=self.process)
        self.mq.channel.start_consuming()

    def process(self, ch, method, props, body):
        self.log.info(method.routing_key + ' received ' + props.correlation_id)
        image = load_image(body)


def main():
    master = SignMaster()


if __name__ == "__main__":
    main()
