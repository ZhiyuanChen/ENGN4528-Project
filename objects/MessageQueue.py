import pika

from globals import *
from .Log import Log


class MessageQueue(object):
    def __init__(self, channel):
        self.channel = channel
        self.host = MQ_HOST
        self.port = MQ_PORT
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(
            host=self.host, port=self.port, virtual_host=MQ_VHOST, credentials=pika.PlainCredentials(MQ_USNM, MQ_PSWD)))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.channel, durable=MQ_DURABLE)
        self.log = Log('message_queue')

    def publish(self, message):
        self.channel.basic_publish(
            exchange='', routing_key=self.channel, body=message, properties=pika.BasicProperties(delivery_mode=MQ_MODE))
        self.log.info(self.queue + ' published ' + message)

