from globals import Master, Vernie, load_message, load_image, OBST_REQUEST


class ObstMaster(Master):
    def __init__(self):
        super(ObstMaster, self).__init__(OBST_REQUEST)
        self.mq.channel.basic_consume(queue=OBST_REQUEST, on_message_callback=self.receive)
        self.mq.channel.start_consuming()

    def receive(self, ch, method, props, body):
        self.log.info('Received message from ' + method.routing_key)
        try:
            code, message, data = load_message(body)
            if code != 200:
                raise Vernie(300, 'Illegal message')
            image = load_image(data)
        except Exception as e:
            self.log.error(e)


def main():
    master = ObstMaster()


if __name__ == "__main__":
    main()
