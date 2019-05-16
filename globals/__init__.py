import configparser
import json
import os

# Initial global variable for config
CONFIG_PATH = os.path.join(os.getcwd(), 'config.ini')
CONFIG = configparser.RawConfigParser()
CONFIG.read(CONFIG_PATH)
# Initial global variables for log
LOG_DIR = os.path.join(os.getcwd(), 'logs')
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
LANE_REQUEST = CONFIG.get('Message Queue', 'Lane Request Queue')
LANE_RESPONSE = CONFIG.get('Message Queue', 'Lane Response Queue')
OBST_REQUEST = CONFIG.get('Message Queue', 'Obstacle Request Queue')
OBST_RESPONSE = CONFIG.get('Message Queue', 'Obstacle Response Queue')
SIGN_REQUEST = CONFIG.get('Message Queue', 'Sign Request Queue')
SIGN_RESPONSE = CONFIG.get('Message Queue', 'Sign Response Queue')


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
