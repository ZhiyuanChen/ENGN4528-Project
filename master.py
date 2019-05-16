import time
import traceback
from globals import Message, MessageQueue, Log
from objects import Image, Truck


class Master(object):
    def __init__(self):
        self.log = Log()
        self.mq = MessageQueue()
