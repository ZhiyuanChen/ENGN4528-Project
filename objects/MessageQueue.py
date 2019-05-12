import os
import configparser
import pika

# Read configuration file
CONFIG_PATH = os.path.join(os.getcwd(), 'config.ini')
CONFIG = configparser.RawConfigParser()
CONFIG.read(CONFIG_PATH)

# Initial global variables for message queue
MQ_HOST = CONFIG.get('Message Queue', 'Host')
MQ_PORT = CONFIG.getint('Message Queue', 'Port')
MQ_VHOST = CONFIG.get('Message Queue', 'Virtual Host')
MQ_USNM = CONFIG.get('Message Queue', 'Username')
MQ_PSWD = CONFIG.get('Message Queue', 'Password')
MQ_DURABLE = CONFIG.getboolean('Message Queue', 'Durable')
MQ_MODE = CONFIG.getint('Message Queue', 'Delivery Mode')
LANE_REQUEST = CONFIG.get('Message Queue', 'Lane Request Queue')
LANE_RESPONSE = CONFIG.get('Message Queue', 'Lane Response Queue')
OBST_REQUEST = CONFIG.get('Message Queue', 'Obstacle Request Queue')
OBST_RESPONSE = CONFIG.get('Message Queue', 'Obstacle Response Queue')
SIGN_REQUEST = CONFIG.get('Message Queue', 'Sign Request Queue')
SIGN_RESPONSE = CONFIG.get('Message Queue', 'Sign Response Queue')


class MessageQueue(object):
    def __init__(self):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(
            host=MQ_HOST, port=MQ_PORT, virtual_host=MQ_VHOST, credentials=pika.PlainCredentials(MQ_USNM, MQ_PSWD)))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=LANE_REQUEST, durable=MQ_DURABLE)
        self.channel.queue_declare(queue=LANE_RESPONSE, durable=MQ_DURABLE)
        self.channel.queue_declare(queue=OBST_REQUEST, durable=MQ_DURABLE)
        self.channel.queue_declare(queue=OBST_RESPONSE, durable=MQ_DURABLE)
        self.channel.queue_declare(queue=SIGN_REQUEST, durable=MQ_DURABLE)
        self.channel.queue_declare(queue=SIGN_RESPONSE, durable=MQ_DURABLE)

    def publish_to_queue(self, routing_key, response_message):
        self.channel.basic_publish(exchange='', routing_key=routing_key, body=response_message,
                                   properties=pika.BasicProperties(delivery_mode=MQ_MODE))
