from lane import *

from globals import Master, Vernie, traceback, load_image, MQ, NN


class LaneMaster(Master):
    def __init__(self):
        super(LaneMaster, self).__init__(MQ.LANE_REQUEST)
        self.model = Lane()
        self.mq.channel.basic_consume(queue=MQ.LANE_REQUEST, on_message_callback=self.process)
        self.mq.channel.start_consuming()

    def process(self, ch, method, props, body):
        self.log.info(method.routing_key + ' received ' + props.correlation_id)
        image = load_image(body)
        image = self.model.process_image(image)


class Lane(object):
    def __init__(self):
        input_tensor = tf.placeholder(dtype=tf.string, shape=[None], name='input_tensor')
        phase_tensor = tf.constant('test', tf.string)
        net = LaneNet()
        tf_vars = tf.global_variables()[:-1]
        saver = tf.train.Saver(tf_vars)
        sess_config = tf.ConfigProto(device_count={'GPU': NN.GPU_AMOUNT})
        sess_config.gpu_options.per_process_gpu_memory_fraction = NN.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = LANE.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'
        sess = tf.Session(config=sess_config)

    @staticmethod
    def process_image(image):
        img_resized = tf.image.resize_images(image, [LANE.IMG_HEIGHT, LANE.IMG_WIDTH], method=tf.image.ResizeMethod.BICUBIC)
        img_casted = tf.cast(img_resized, tf.float32)
        return tf.subtract(img_casted, NN.VGG_MEAN)


def main():
    master = LaneMaster()


if __name__ == "__main__":
    main()
