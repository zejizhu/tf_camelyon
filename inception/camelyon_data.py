"""Small library that points to the flowers data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from inception.dataset import Dataset

class CamelyonData(Dataset):
  """Flowers data set."""

  def __init__(self, subset):
    super(CamelyonData, self).__init__('Camelyon', subset)

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return 2

  def num_examples_per_epoch(self):
    """Returns the number of examples in the data subset."""
    if self.subset == 'train':
      #return 809032
      return 667300
    if self.subset == 'validation':
      return 8000
      #return 10000
    if self.subset == 'test':
      return 2370610


  def download_message(self):
    """Instruction to download and extract the tarball from Flowers website."""

    print('Failed to find any Camelyon %s files'% self.subset)
    print('')
    print('If you have already downloaded and processed the data, then make '
          'sure to set --data_dir to point to the directory containing the '
          'location of the sharded TFRecords.\n')
    print('Please see README.md for instructions on how to build '
          'the flowers dataset using download_and_preprocess_flowers.\n')
