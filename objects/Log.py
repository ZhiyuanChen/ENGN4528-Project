import logging
import logging.handlers
import os
import configparser

CONFIG_PATH = os.path.join(os.getcwd(), 'config.ini')
CONFIG = configparser.RawConfigParser()
CONFIG.read(CONFIG_PATH)
# Initial global variables for log
LOG_LVL = CONFIG.get('Log', 'Level')
LOG_WHEN = CONFIG.get('Log', 'When')
LOG_INTV = CONFIG.getint('Log', 'Interval')
LOG_MAXC = CONFIG.getint('Log', 'Max Counter')
LOG_FMT = CONFIG.get('Log', 'Format')


class Log(object):
    def __init__(self, file=CONFIG.get('Log', 'File')):
        log_file = os.path.join(os.getcwd(), 'logs', file)
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
