import pika

from globals import *
from .Log import Log


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
        self.log = Log('message_queue')

    def publish(self, queue, message):
        self.channel.basic_publish(
            exchange='', routing_key=queue, body=message, properties=pika.BasicProperties(delivery_mode=MQ_MODE))
        self.log.info(queue + ' published ' + message)

    def host(self):
        return MQ_HOST

    def port(self):
        return MQ_PORT
