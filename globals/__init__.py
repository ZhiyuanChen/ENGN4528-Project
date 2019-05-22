import configparser
import json
import logging
import logging.handlers
import os
import traceback
import time

import cv2
import numpy as np
import pika
from easydict import EasyDict as edict


__LOG = edict()
__MQ = edict()
__CONCURRENT = edict()
__NN = edict()
LOG = __LOG
MQ = __MQ
CONCURRENT = __CONCURRENT
NN = __NN
# Initial global variable for config
PROJECT_PATH = 'D:\\OneDrive\\OneDrive - Australian National University\\COMP\\4528\\project'
CONFIG_PATH = os.path.join(PROJECT_PATH, 'config.ini')
CONFIG = configparser.RawConfigParser()
CONFIG.read(CONFIG_PATH)
# Initial global variables for log
__LOG.DIR = os.path.join(PROJECT_PATH, 'logs')
__LOG.LVL = CONFIG.get('Log', 'Level')
__LOG.WHEN = CONFIG.get('Log', 'When')
__LOG.INTV = CONFIG.getint('Log', 'Interval')
__LOG.MAXC = CONFIG.getint('Log', 'Max Counter')
__LOG.FMT = CONFIG.get('Log', 'Format')
# Initial global variables for message queue
__MQ.HOST = CONFIG.get('Message Queue', 'Host')
__MQ.PORT = CONFIG.getint('Message Queue', 'Port')
__MQ.VHOST = CONFIG.get('Message Queue', 'Virtual Host')
__MQ.USNM = CONFIG.get('Message Queue', 'Username')
__MQ.PSWD = CONFIG.get('Message Queue', 'Password')
__MQ.DURABLE = CONFIG.getboolean('Message Queue', 'Durable')
__MQ.MODE = CONFIG.getint('Message Queue', 'Delivery Mode')
__MQ.COMP_REQUEST = CONFIG.get('Message Queue', 'Comprehensive Request Queue')
__MQ.COMP_RESPONSE = CONFIG.get('Message Queue', 'Comprehensive Response Queue')
__MQ.LANE_REQUEST = CONFIG.get('Message Queue', 'Lane Request Queue')
__MQ.LANE_RESPONSE = CONFIG.get('Message Queue', 'Lane Response Queue')
__MQ.OBST_REQUEST = CONFIG.get('Message Queue', 'Obstacle Request Queue')
__MQ.OBST_RESPONSE = CONFIG.get('Message Queue', 'Obstacle Response Queue')
__MQ.SIGN_REQUEST = CONFIG.get('Message Queue', 'Sign Request Queue')
__MQ.SIGN_RESPONSE = CONFIG.get('Message Queue', 'Sign Response Queue')
__MQ.PREFETCH_NUM = int(CONFIG.get('Message Queue', 'Consume Number'))
# Initial global variables for concurrency
__CONCURRENT.MAX_WORKER = int(CONFIG.get('Concurrency', 'Max Workers'))
# Initial global variables for neural network
__NN.VGG_MEAN = np.array([123.68, 116.779, 103.939])
__NN.CPU_AMOUNT = int(CONFIG.get('Neural Network', 'CPU Amount'))
__NN.GPU_AMOUNT = int(CONFIG.get('Neural Network', 'GPU Amount'))
__NN.LANE_WEIGHTS_PATH = os.path.join(PROJECT_PATH, 'lane/model')
__NN.OBST_WEIGHTS_PATH = os.path.join(PROJECT_PATH, 'obstacle/model/model.h5')
__NN.SIGN_WEIGHTS_PATH = os.path.join(PROJECT_PATH, 'sign/model')


def load_message(message):
    try:
        message = json.loads(message)
        return message['code'], message['msg'], message['data']
    except Exception:
        raise Vernie(306, 'Failed to load message', traceback.format_exc())


def load_image(data):
    try:
        return cv2.imdecode(np.fromstring(data, np.uint8), 1)
    except Exception:
        raise Vernie(307, 'Failed to load image', traceback.format_exc())


class Master(object):
    def __init__(self, channel=None):
        self.mq = MessageQueue()
        self.mq.channel.basic_qos(prefetch_count=MQ.PREFETCH_NUM)
        if channel is not None:
            self.log = Log(channel)
            self.log.info('---------------------------------------')
            self.log.info('Listening ' + channel + ' on ' + self.mq.host() + ':' + str(self.mq.port()))
            self.log.info('---------------------------------------')
        else:
            self.log = Log('master')
        # This parameter MUST be overwritten in subclass
        self.queue = None

    # This function MUST be overwrite in subclass, do NOT call this function
    def process(self, ch, method, props, body):
        self.log.info(method.routing_key + ' received message')
        image = load_image(body)


class Message(object):
    def __init__(self, code, message, data):
        self.code = code
        self.message = message
        self.data = data

    def json(self):
        return json.dumps({'code': self.code, 'msg': self.message, 'data': self.data})


class MessageQueue(object):
    def __init__(self):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(
            host=MQ.HOST, port=MQ.PORT, virtual_host=MQ.VHOST, credentials=pika.PlainCredentials(MQ.USNM, MQ.PSWD)))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=MQ.COMP_REQUEST, durable=MQ.DURABLE)
        self.channel.queue_declare(queue=MQ.COMP_RESPONSE, durable=MQ.DURABLE)
        self.channel.queue_declare(queue=MQ.LANE_REQUEST, durable=MQ.DURABLE)
        self.channel.queue_declare(queue=MQ.LANE_RESPONSE, durable=MQ.DURABLE)
        self.channel.queue_declare(queue=MQ.OBST_REQUEST, durable=MQ.DURABLE)
        self.channel.queue_declare(queue=MQ.OBST_RESPONSE, durable=MQ.DURABLE)
        self.channel.queue_declare(queue=MQ.SIGN_REQUEST, durable=MQ.DURABLE)
        self.channel.queue_declare(queue=MQ.SIGN_RESPONSE, durable=MQ.DURABLE)
        self.log = Log('message_queue')

    def publish(self, queue, message):
        self.channel.basic_publish(exchange='', routing_key=queue, body=message)
        self.log.info('Publish message to: ' + queue)

    @staticmethod
    def host():
        return MQ.HOST

    @staticmethod
    def port():
        return MQ.PORT


class Log(object):
    def __init__(self, file=CONFIG.get('Log', 'File')):
        log_file = os.path.join(LOG.DIR, file)
        logging.basicConfig()
        self.logger = logging.getLogger()
        self.logger.setLevel(LOG.LVL)
        self.handler = logging.handlers.TimedRotatingFileHandler(filename=log_file, when=LOG.WHEN, interval=LOG.INTV)
        self.handler.suffix = '%Y-%m-%d.log'
        self.handler.setLevel(LOG.LVL)
        self.handler.setFormatter(logging.Formatter(LOG.FMT))
        self.logger.addHandler(self.handler)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)


class Vernie(Exception):
    def __init__(self, code, message, detail=None):
        self.code = code
        self.message = message
        self.detail = detail
