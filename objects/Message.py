import json


class Message(object):
    def __init__(self, code, message, data):
        self.code = code
        self.message = message
        self.data = data

    def json(self):
        return json.dumps({'code': self.code, 'msg': self.message, 'data': self.data})
