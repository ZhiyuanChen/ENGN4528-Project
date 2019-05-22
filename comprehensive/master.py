from globals import Master, Vernie, traceback, load_image, MQ
from objects import Truck


class CompMaster(Master):
    def __init__(self):
        super(CompMaster, self).__init__(MQ.SIGN_REQUEST)
        self.mq.channel.basic_consume(queue=MQ.COMP_REQUEST, on_message_callback=self.process)
        self.mq.channel.start_consuming()
        self.truck = Truck()

    def process(self, ch, method, props, body):
        self.log.info(method.routing_key + ' received message')
        image = load_image(body)
        self.truck.init(image)
        self.mq.publish()


def main():
    master = CompMaster()


if __name__ == "__main__":
    main()
