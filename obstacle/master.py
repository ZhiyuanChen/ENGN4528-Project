import cv2
import numpy as np
import configparser
import os
import sys
sys.path.append("..")
from objects.Log import Log
from objects.MessageQueue import MessageQueue
from global_functions import load_message


CONFIG_PATH = os.path.join(os.getcwd(), 'config.ini')
CONFIG = configparser.RawConfigParser()
CONFIG.read(CONFIG_PATH)
OBST_REQUEST = CONFIG.get('Message Queue', 'Obstacle Request Queue')
PREFETCH_NUM = int(CONFIG.get('Concurrency', 'Consume Number'))


class Master(object):
    def __init__(self):
        self.log = Log('obstacle_request')
        self.mq = MessageQueue()
        self.mq.channel.basic_qos(prefetch_count=PREFETCH_NUM)
        self.mq.channel.basic_consume(self.receive, queue=OBST_REQUEST)
        self.log.info('------------------------------------')
        self.log.info('HOST: ' + self.mq.host() + 'PORT: ' + str(self.mq.port()) + 'QUEUE: ' + OBST_REQUEST + 'Ready to consume')
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
