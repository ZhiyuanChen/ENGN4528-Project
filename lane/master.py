from lane import *

from globals import Master, Vernie, traceback, load_image, MQ, NN


def test_lanenet(weights_path, use_gpu, image_list, batch_size, save_dir):
    test_dataset = lanenet_data_processor_test.DataSet(image_path, batch_size)
    input_tensor = tf.placeholder(dtype=tf.string, shape=[None], name='input_tensor')
    imgs = tf.map_fn(test_dataset.process_img, input_tensor, dtype=tf.float32)
    phase_tensor = tf.constant('test', tf.string)

    net = lanenet_merge_model.LaneNet()
    binary_seg_ret, instance_seg_ret = net.test_inference(imgs, phase_tensor, 'lanenet_loss')
    initial_var = tf.global_variables()
    final_var = initial_var[:-1]
    saver = tf.train.Saver(final_var)
    # Set sess configuration
    sess_config = tf.ConfigProto(device_count={'GPU': 1})

    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=sess_config)
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        saver.restore(sess=sess, save_path=weights_path)
        paths = test_dataset.next_batch()
        image_list, existence_output = sess.run([binary_seg_ret, instance_seg_ret], feed_dict={input_tensor: paths})
        image = image_list[0]
    sess.close()
    return


class LaneMaster(Master):
    def __init__(self):
        super(LaneMaster, self).__init__(MQ.LANE_REQUEST)
        self.model = Lane()
        self.mq.channel.basic_consume(queue=MQ.LANE_REQUEST, on_message_callback=self.process)
        self.mq.channel.start_consuming()

    def process(self, ch, method, props, body):
        self.log.info(method.routing_key + ' received message')
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
