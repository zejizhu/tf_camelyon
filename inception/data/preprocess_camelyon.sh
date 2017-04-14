#!/bin/sh
set -e

TOP_DIR="/home1/zhuzj/dataset/c16_test_B2/"
DIR_RAW_DATA="$TOP_DIR/input_sample/*"
FILE_PATH_LABEL="$TOP_DIR/labels.txt"

# Create the output and temporary directories.
#DATA_DIR="${1%/}"
DATA_DIR="$TOP_DIR"
SCRATCH_DIR="${DATA_DIR}/raw-data"
mkdir -p "${DATA_DIR}"
mkdir -p "${SCRATCH_DIR}"
# http://stackoverflow.com/questions/59895/getting-the-source-directory-of-a-bash-script-from-within
#WORK_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORK_DIR="./data/"
# Download the flowers data.
#DATA_URL="http://download.tensorflow.org/example_images/flower_photos.tgz"
CURRENT_DIR=$(pwd)
#TARBALL="flower_photos.tgz"
#if [ ! -f ${TARBALL} ]; then
#  echo "Downloading flower data set."
#  curl -o ${DATA_DIR}/${TARBALL} "${DATA_URL}"
#else
#  echo "Skipping download of flower data."
#fi

# Note the locations of the train and validation data.
TRAIN_DIRECTORY="${SCRATCH_DIR}/train"
VALIDATION_DIRECTORY="${SCRATCH_DIR}/validation"

# Expands the data into the flower_photos/ directory and rename it as the
# train directory.
#tar xf ${DATA_DIR}/flower_photos.tgz
rm -rf "${TRAIN_DIRECTORY}" "${VALIDATION_DIRECTORY}"
mkdir -p "${TRAIN_DIRECTORY}"
mv ${DIR_RAW_DATA} ${TRAIN_DIRECTORY}
#printf $strs
#mv "${DIR_RAW_DATA}" "${TRAIN_DIRECTORY}"
#`$strs`

# Generate a list of 5 labels: daisy, dandelion, roses, sunflowers, tulips
#LABELS_FILE="${SCRATCH_DIR}/labels.txt"
LABELS_FILE=$FILE_PATH_LABEL
ls -1 "${TRAIN_DIRECTORY}" | grep -v 'LICENSE' | sed 's/\///' | sort > "${LABELS_FILE}"

# Generate the validation data set.
while read LABEL; do
  VALIDATION_DIR_FOR_LABEL="${VALIDATION_DIRECTORY}/${LABEL}"
  TRAIN_DIR_FOR_LABEL="${TRAIN_DIRECTORY}/${LABEL}"

  # Move the first randomly selected 5000 images to the validation set.
  mkdir -p "${VALIDATION_DIR_FOR_LABEL}"
  VALIDATION_IMAGES=$(ls -1 "${TRAIN_DIR_FOR_LABEL}" | shuf | head -2000)
  for IMAGE in ${VALIDATION_IMAGES}; do
    mv -f "${TRAIN_DIRECTORY}/${LABEL}/${IMAGE}" "${VALIDATION_DIR_FOR_LABEL}"
  done
done < "${LABELS_FILE}"

# Build the TFRecords version of the image data.
cd "${CURRENT_DIR}"
BUILD_SCRIPT="${WORK_DIR}/build_image_data.py"
OUTPUT_DIRECTORY="${DATA_DIR}/TFRecords/"
mkdir -p "${OUTPUT_DIRECTORY}"
python "${BUILD_SCRIPT}" \
  --train_directory="${TRAIN_DIRECTORY}" \
  --validation_directory="${VALIDATION_DIRECTORY}" \
  --output_directory="${OUTPUT_DIRECTORY}" \
  --labels_file="${LABELS_FILE}" \
  --train_shards=128 \
  --validation_shards=32 \
  --num_threads=8
