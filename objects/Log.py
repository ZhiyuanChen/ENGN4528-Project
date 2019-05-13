import logging
import logging.handlers

from globals import *


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
