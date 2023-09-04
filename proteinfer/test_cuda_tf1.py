import tensorflow as tf

# Ensure that we're using TensorFlow 1.15
if not tf.__version__.startswith('1.15'):
    raise ValueError('This code requires TensorFlow V1.15!')

# Check TensorFlow version
print("TensorFlow Version:", tf.__version__)

# Check if TensorFlow is built with CUDA (GPU support)
print("Is TensorFlow built with CUDA:", tf.test.is_built_with_cuda())

# Check if GPU device is available
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

# List available GPUs
from tensorflow.python.client import device_lib
print("List of available devices:", [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print (sess.run(c))