from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def init():
    '''DIR'''
    tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                               """Directory where to write event logs """
                               """and checkpoint.""")
    tf.app.flags.DEFINE_string('eval_dir', '/tmp/imagenet_eval',
                               """Directory where to write event logs.""")
    tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/imagenet_train',
                               """Directory where to read model checkpoints.""")

    tf.app.flags.DEFINE_string('csv_dir', '/home1/zhuzj/dataset/camelyon16_B2/csv_out',
                               """Directory where to read model checkpoints.""")

    tf.app.flags.DEFINE_string('test_dir', '/tmp/imagenet_eval',
                               """Directory where to write event logs.""")

    tf.app.flags.DEFINE_string('data_dir', '/tmp/mydata',
                               """Path to the processed data, i.e. """
                               """TFRecord of Example protos.""")

    tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                               """If specified, restore this pretrained model """
                               """before beginning any training.""")

    '''PARAM'''
    tf.app.flags.DEFINE_integer('max_steps', 10000000,
                                """Number of batches to run.""")
    tf.app.flags.DEFINE_string('subset', 'train',
                               """Either 'train' or 'validation'.""")

    # Flags governing the hardware employed for running TensorFlow.
    tf.app.flags.DEFINE_integer('num_gpus', 1,
                                """How many GPUs to use.""")
    tf.app.flags.DEFINE_boolean('log_device_placement', False,
                                """Whether to log device placement.""")

    # Flags governing the type of training.
    tf.app.flags.DEFINE_boolean('fine_tune', False,
                                """If set, randomly initialize the final layer """
                                """of weights in order to train the network on a """
                                """new task.""")


    # **IMPORTANT**
    # Please note that this learning rate schedule is heavily dependent on the
    # hardware architecture, batch size and any changes to the model architecture
    # specification. Selecting a finely tuned learning rate schedule is an
    # empirical process that requires some experimentation. Please see README.md
    # more guidance and discussion.
    #
    # With 8 Tesla K40's and a batch size = 256, the following setup achieves
    # precision@1 = 73.5% after 100 hours and 100K steps (20 epochs).
    # Learning rate decay factor selected from http://arxiv.org/abs/1404.5997.
    tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                              """Initial learning rate.""")
    tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0,
                              """Epochs after which learning rate decays.""")
    tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16,
                              """Learning rate decay factor.""")



    # Flags governing the frequency of the eval.
    tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                                """How often to run the eval.""")
    tf.app.flags.DEFINE_boolean('run_once', False,
                                """Whether to run eval only once.""")

    # Flags governing the data used for the eval.
    tf.app.flags.DEFINE_integer('num_examples', 50000,
                                """Number of examples to run. Note that the eval """
                                """ImageNet dataset contains 50000 examples.""")

    return 0

