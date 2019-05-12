import os
import configparser
import pika
from objects import Log





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
        Log.info(routing_key + ' published ' + response_message)
