"""A binary to evaluate Inception on the flowers data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inception import inception_test
from inception.camelyon_data import CamelyonData

FLAGS = tf.app.flags.FLAGS


def main(unused_argv=None):
  dataset = CamelyonData(subset=FLAGS.subset)
  assert dataset.data_files()
  if tf.gfile.Exists(FLAGS.test_dir):
    tf.gfile.DeleteRecursively(FLAGS.test_dir)
  tf.gfile.MakeDirs(FLAGS.test_dir)
  inception_test.test(dataset)


if __name__ == '__main__':
  tf.app.run()
