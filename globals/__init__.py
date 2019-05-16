import configparser
import json
import os

import cv2
import numpy as np

from objects.Log import Log
from objects.MessageQueue import MessageQueue

# Initial global variable for config
CONFIG_PATH = os.path.join(os.path.dirname(os.getcwd()), 'config.ini')
CONFIG = configparser.RawConfigParser()
CONFIG.read(CONFIG_PATH)
# Initial global variables for log
LOG_DIR = os.path.join(os.path.dirname(os.getcwd()), 'logs')
LOG_LVL = CONFIG.get('Log', 'Level')
LOG_WHEN = CONFIG.get('Log', 'When')
LOG_INTV = CONFIG.getint('Log', 'Interval')
LOG_MAXC = CONFIG.getint('Log', 'Max Counter')
LOG_FMT = CONFIG.get('Log', 'Format')
# Initial global variables for message queue
PREFETCH_NUM = int(CONFIG.get('Concurrency', 'Consume Number'))
MQ_HOST = CONFIG.get('Message Queue', 'Host')
MQ_PORT = CONFIG.getint('Message Queue', 'Port')
MQ_VHOST = CONFIG.get('Message Queue', 'Virtual Host')
MQ_USNM = CONFIG.get('Message Queue', 'Username')
MQ_PSWD = CONFIG.get('Message Queue', 'Password')
MQ_DURABLE = CONFIG.getboolean('Message Queue', 'Durable')
MQ_MODE = CONFIG.getint('Message Queue', 'Delivery Mode')
COMP_REQUEST = CONFIG.get('Message Queue', 'Comprehensive Request Queue')
COMP_RESPONSE = CONFIG.get('Message Queue', 'Comprehensive Response Queue')
LANE_REQUEST = CONFIG.get('Message Queue', 'Lane Request Queue')
LANE_RESPONSE = CONFIG.get('Message Queue', 'Lane Response Queue')
OBST_REQUEST = CONFIG.get('Message Queue', 'Obstacle Request Queue')
OBST_RESPONSE = CONFIG.get('Message Queue', 'Obstacle Response Queue')
SIGN_REQUEST = CONFIG.get('Message Queue', 'Sign Request Queue')
SIGN_RESPONSE = CONFIG.get('Message Queue', 'Sign Response Queue')
MAX_WORKER = int(CONFIG.get('Concurrency', 'Max Workers'))


def load_message(message):
    try:
        message = json.loads(message)
    except Exception:
        pass
    return message['code'], message['message'], message['data']


def get_queue(queue):
    try:
        if queue == 'lane':
            return LANE_REQUEST
        elif queue == 'obstacle':
            return OBST_REQUEST
        elif queue == 'sign':
            return SIGN_REQUEST
        else:
            raise Exception
    except Exception:
        pass


class Master(object):
    def __init__(self, channel):
        self.log = Log(channel)
        self.mq = MessageQueue()
        self.mq.channel.basic_qos(prefetch_count=PREFETCH_NUM)
        self.log.info('------------------------------------')
        self.log.info('Listening ' + channel + ' on ' + self.mq.host() + ':' + str(self.mq.port()))
        self.log.info('------------------------------------')

    def receive(self, ch, method, props, body):
        self.log.info('************ Received Request ************')
        try:
            code, message, image_dict = load_message(body)
            if code != 200:
                raise Exception
            windshield = cv2.imdecode(np.fromstring(image_dict['windshield'], np.uint8), 1)

        except Exception:
            pass


class CVException(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message
