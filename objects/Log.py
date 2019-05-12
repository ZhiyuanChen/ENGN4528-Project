import logging
import logging.handlers
import os
import configparser

CONFIG_PATH = os.path.join(os.getcwd(), 'config.ini')
CONFIG = configparser.RawConfigParser()
CONFIG.read(CONFIG_PATH)
# Initial global variables for log
LOG_FILE = os.path.join(os.getcwd(), 'logs',  CONFIG.get('Log', 'File'))
LOG_LVL = CONFIG.get('Log', 'Level')
LOG_WHEN = CONFIG.get('Log', 'When')
LOG_INTV = CONFIG.getint('Log', 'Interval')
LOG_MAXC = CONFIG.getint('Log', 'Max Counter')
LOG_FMT = CONFIG.get('Log', 'Format')


class Log(object):
    def __init__(self):
        logging.basicConfig()
        self.logger = logging.getLogger()
        self.logger.setLevel(LOG_LVL)
        self.handler = logging.handlers.TimedRotatingFileHandler(filename=LOG_FILE, when=LOG_WHEN, interval=LOG_INTV)
        self.handler.suffix = '%Y-%m-%d.log'
        self.handler.setLevel(LOG_LVL)
        self.handler.setFormatter(logging.Formatter(LOG_FMT))
        self.logger.addHandler(self.handler)

    def info(self, message):
        self.logger.info(message)
