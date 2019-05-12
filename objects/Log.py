import logging
import logging.handlers


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
