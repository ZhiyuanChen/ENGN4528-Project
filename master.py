import time
import traceback
from objects.Log import Log
from objects.Truck import Truck
from objects.Image import Image
from objects.MessageQueue import MessageQueue
from objects.Message import Message


class Master(object):
    log = Log()
    log.info('message')
