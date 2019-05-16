import time
import traceback

from globals import Master, COMP_REQUEST

from objects.Message import Message
from objects.Image import Image
from objects.Truck import Truck


class CompMaster(Master):
    def __init__(self):
        super(CompMaster, self).__init__(COMP_REQUEST)
        self.mq.channel.basic_consume(queue=COMP_REQUEST, on_message_callback=self.receive)
        self.mq.channel.start_consuming()
