import time
import traceback
from objects.Log import Log
from objects.MessageQueue import MessageQueue
from objects.Message import Message
from objects.Image import Image
from objects.Truck import Truck


class Master(object):
    def __init__(self):
        self.log = Log()
        self.mq = MessageQueue()
