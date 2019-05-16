import time
import traceback
from globals import MessageQueue, Log
from objects.Message import Message
from objects.Image import Image
from objects.Truck import Truck


class Master(object):
    def __init__(self):
        self.log = Log()
        self.mq = MessageQueue()
