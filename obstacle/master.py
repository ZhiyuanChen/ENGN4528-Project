import cv2
import numpy as np

from globals import load_message, OBST_REQUEST, PREFETCH_NUM
from objects.Log import Log
from objects.MessageQueue import MessageQueue


class Master(object):
    def __init__(self):
        self.log = Log('obstacle_request')
        self.mq = MessageQueue()
        self.mq.channel.basic_qos(prefetch_count=PREFETCH_NUM)
        self.mq.channel.basic_consume(queue=OBST_REQUEST, on_message_callback=self.receive)
        self.log.info('------------------------------------')
        self.log.info('HOST: ' + self.mq.host() + ' PORT: ' + str(
            self.mq.port()) + ' QUEUE: ' + OBST_REQUEST + ' Ready to consume')
        self.log.info('------------------------------------')
        self.mq.channel.start_consuming()

    def receive(self, ch, method, props, body):
        self.log.info('************ Received Request ************')
        try:
            code, message, image_dict = load_message(body)
            if code != 200:
                raise Exception
            windshield = cv2.imdecode(np.fromstring(image_dict['windshield'], np.uint8), 1)
        except Exception:
            pass


def main():
    master = Master()


if __name__ == "__main__":
    main()
