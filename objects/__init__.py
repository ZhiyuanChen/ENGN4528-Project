import os
import glob
import configparser

modules = glob.glob(os.path.dirname(__file__)+"/*.py")
__all__ = [os.path.basename(f)[:-3] for f in modules]
CONFIG_PATH = os.path.join(os.getcwd(), 'config.ini')
print(CONFIG_PATH)
print(os.path.exists(CONFIG_PATH))
CONFIG = configparser.RawConfigParser()
CONFIG.read(CONFIG_PATH)

LOG_FILE = CONFIG.get('Log', 'File')
LOG_LVL = CONFIG.get('Log', 'Level')
LOG_WHEN = CONFIG.get('Log', 'When')
LOG_INTV = CONFIG.getint('Log', 'Interval')
LOG_MAXC = CONFIG.getint('Log', 'Max Counter')
LOG_FMT = CONFIG.get('Log', 'Format')

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
