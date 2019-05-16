from globals import Master, load_message, load_image, COMP_REQUEST


class CompMaster(Master):
    def __init__(self):
        super(CompMaster, self).__init__(COMP_REQUEST)
        self.mq.channel.basic_consume(queue=COMP_REQUEST, on_message_callback=self.receive)
        self.mq.channel.start_consuming()

    def receive(self, ch, method, props, body):
        self.log.info('Received message from ' + method.routing_key)
        try:
            code, message, data = load_message(body)
            if code != 200:
                raise Exception
            image = load_image(data)
        except Exception:
            pass
