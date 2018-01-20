#!/usr/bin/env sh
set -e

FILE_NAME=speech_commands_v0.01.tar.gz
URL=http://download.tensorflow.org/data/$FILE_NAME
DATASET_FOLDER=dataset/speech_commands

echo "downloading $URL...\n"
wget -O dataset/$FILE_NAME $URL

echo "extracting $FILE_NAME..."
TEMP_DIRECTORY=$DATASET_FOLDER/audio
mkdir -p $TEMP_DIRECTORY
tar -xzf dataset/$FILE_NAME -C $TEMP_DIRECTORY

echo "splitting the dataset into train, validation and test sets..."
python $DATASET_FOLDER/split_dataset.py $DATASET_FOLDER

echo "done"
