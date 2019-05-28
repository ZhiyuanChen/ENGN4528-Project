from globals import Master, Vernie, traceback, load_image, MQ, cv2
from objects import Truck


class CompMaster(Master):
    def __init__(self):
        super(CompMaster, self).__init__(MQ.COMP_REQUEST)
        self.truck = Truck()
        self.mq.channel.basic_consume(queue=MQ.COMP_REQUEST, on_message_callback=self.process)
        self.mq.channel.start_consuming()

    def process(self, ch, method, props, body):
        try:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            self.log.info(method.routing_key + ' received message')
            image = load_image(body)
            image = self.truck.dashboard(image)
            data = cv2.imencode('.jpg', image)[1].tostring()
            self.mq.publish(MQ.COMP_RESPONSE, data)
        except Exception as err:
            self.log.error(err)

    @staticmethod
    def draw_result(image, truck):
        pass


def main():
    master = CompMaster()


if __name__ == "__main__":
    main()
