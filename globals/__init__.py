import configparser
import json
import logging
import logging.handlers
import os
import uuid
import cv2
import numpy as np
import pika

# Initial global variable for config
CONFIG_PATH = os.path.join('D:\\OneDrive\\OneDrive - Australian National University\\COMP\\4528\\project', 'config.ini')
CONFIG = configparser.RawConfigParser()
CONFIG.read(CONFIG_PATH)
# Initial global variables for log
LOG_DIR = os.path.join('D:\\OneDrive\\OneDrive - Australian National University\\COMP\\4528\\project', 'logs')
LOG_LVL = CONFIG.get('Log', 'Level')
LOG_WHEN = CONFIG.get('Log', 'When')
LOG_INTV = CONFIG.getint('Log', 'Interval')
LOG_MAXC = CONFIG.getint('Log', 'Max Counter')
LOG_FMT = CONFIG.get('Log', 'Format')
# Initial global variables for message queue
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
# Initial global variables for concurrency
PREFETCH_NUM = int(CONFIG.get('Concurrency', 'Consume Number'))
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
        self.log.info('---------------------------------------')
        self.log.info('Listening ' + channel + ' on ' + self.mq.host() + ':' + str(self.mq.port()))
        self.log.info('---------------------------------------')

    def receive(self, ch, method, props, body):
        self.log.info('Received message')
        try:
            code, message, image_dict = load_message(body)
            if code != 200:
                raise Exception
            windshield = cv2.imdecode(np.fromstring(image_dict['windshield'], np.uint8), 1)

        except Exception:
            pass


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
            host=MQ_HOST, port=MQ_PORT, virtual_host=MQ_VHOST, credentials=pika.PlainCredentials(MQ_USNM, MQ_PSWD)))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=COMP_REQUEST, durable=MQ_DURABLE)
        self.channel.queue_declare(queue=COMP_RESPONSE, durable=MQ_DURABLE)
        self.channel.queue_declare(queue=LANE_REQUEST, durable=MQ_DURABLE)
        self.channel.queue_declare(queue=LANE_RESPONSE, durable=MQ_DURABLE)
        self.channel.queue_declare(queue=OBST_REQUEST, durable=MQ_DURABLE)
        self.channel.queue_declare(queue=OBST_RESPONSE, durable=MQ_DURABLE)
        self.channel.queue_declare(queue=SIGN_REQUEST, durable=MQ_DURABLE)
        self.channel.queue_declare(queue=SIGN_RESPONSE, durable=MQ_DURABLE)
        self.log = Log('message_queue')

    def publish(self, queue, message, callback_queue, corr_id=str(uuid.uuid4())):
        self.channel.basic_publish(
            exchange='', routing_key=queue, body=message, properties=
            pika.BasicProperties(delivery_mode=MQ_MODE), reply_to=callback_queue, correlation_id=corr_id)
        self.log.info('Publish message to: ' + queue)

    def host(self):
        return MQ_HOST

    def port(self):
        return MQ_PORT


class Log(object):
    def __init__(self, file=CONFIG.get('Log', 'File')):
        log_file = os.path.join(LOG_DIR, file)
        logging.basicConfig()
        self.logger = logging.getLogger()
        self.logger.setLevel(LOG_LVL)
        self.handler = logging.handlers.TimedRotatingFileHandler(filename=log_file, when=LOG_WHEN, interval=LOG_INTV)
        self.handler.suffix = '%Y-%m-%d.log'
        self.handler.setLevel(LOG_LVL)
        self.handler.setFormatter(logging.Formatter(LOG_FMT))
        self.logger.addHandler(self.handler)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)


class CVException(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message
