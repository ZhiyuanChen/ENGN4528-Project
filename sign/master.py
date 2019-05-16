from globals import Master, Vernie, traceback, load_message, load_image, SIGN_REQUEST


class SignMaster(Master):
    def __init__(self):
        super(SignMaster, self).__init__(SIGN_REQUEST)
        self.mq.channel.basic_consume(queue=SIGN_REQUEST, on_message_callback=self.receive)
        self.mq.channel.start_consuming()

    def receive(self, ch, method, props, body):
        self.log.info('Received message from ' + method.routing_key)
        try:
            code, message, data = load_message(body)
            if code != 200:
                raise Exception
            image = load_image(data)
        except Vernie:
            raise Vernie(300, 'Illegal message')
        except Exception:
            raise Vernie(301, 'Failed to receive message', traceback.format_exc())


def main():
    master = SignMaster()


if __name__ == "__main__":
    main()
