import configparser
import os
import time
from concurrent import futures

import numpy as np
from mss import mss

from objects.Image import Image
from objects.Log import Log
from objects.Message import Message
from objects.MessageQueue import MessageQueue

CONFIG_PATH = os.path.join(os.getcwd(), 'config.ini')
CONFIG = configparser.RawConfigParser()
CONFIG.read(CONFIG_PATH)
MAX_WORKER = int(CONFIG.get('Concurrency', 'Max Workers'))
LANE_REQUEST = CONFIG.get('Message Queue', 'Lane Request Queue')
OBST_REQUEST = CONFIG.get('Message Queue', 'Obstacle Request Queue')
SIGN_REQUEST = CONFIG.get('Message Queue', 'Sign Request Queue')


class Capture(object):
    def __init__(self):
        self.log = Log('capture')
        self.mq = MessageQueue()
        self.routing_key_list = [LANE_REQUEST, OBST_REQUEST, SIGN_REQUEST]
        self.thread_pool = futures.ThreadPoolExecutor(max_workers=MAX_WORKER)
        self.screen_shot = None

    def capture(self):
        self.screen_shot = Image(np.array(mss().grab({"top": 40, "left": 0, "width": 1280, "height": 720})))

    def publish(self, routing_key, data):
        self.mq.publish(routing_key, Message(200, 'success', data))

    def publish_concurrent(self):
        windshield_json = self.screen_shot.windshield_json()
        try:
            for routing_key in self.routing_key_list:
                self.thread_pool.submit(self.publish, routing_key, windshield_json)
        except Exception as err:
            self.log.error(err)
            raise Exception(824, '试卷下载失败')

    def process(self):
        t = time.time()
        self.capture()
        self.publish_concurrent()
        print(time.time()-t)


def main():
    capture = Capture()
    while(True):
        capture.process()


if __name__ == "__main__":
    main()
