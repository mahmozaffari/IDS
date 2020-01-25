import numpy as np
from AnomalyDetector import AD
import argparse
import os
import math


def get_args():
    parser = argparse.ArgumentParser(description='Options to run the Anomaly Detector')
    parser.add_argument('-d', '--dataset', type=str, default='', help='set the name of dataset.')
    parser.add_argument('--data_path', type=str, default='', help='set the file path for training data.')
    parser.add_argument('--label_path', type=str, default='', help='set the file path for training labels.')
    parser.add_argument('--train_size', type=int, default=50000, help='set the training size.')
    parser.add_argument('-b', '--batch_size', type=int, default=1000, help='set the batch.')
    #parser.add_argument('-d', '--detector', type=str, default='', help='set the detector')
    parser.add_argument('-o', '--output_folder', type=str, default='', help='set the output folder path.')
    parser.add_argument('--limit', type=int, default=None, help='set the batch')
    parser.add_argument('--alpha', type=float, default=0.05, help='set the significance level of ODIT.')
    parser.add_argument('--k', type=int, default=1, help='set the number of nearest neighbors in kNN.')
    parser.add_argument('--overwrite', type=bool, default=True, help='set the overwrite flag (True/False).')
    parser.add_argument('--debug', type=bool, default=False, help='set the debugging flag.')
    return parser.parse_args()



args = get_args()

dataset_name = args.dataset
data_path = args.data_path
label_path = args.label_path
train_size = args.train_size
batch_size = args.batch_size
limit = args.limit
if limit is None:
    limit = math.inf

overwrite_flag = args.overwrite
output_path = os.path.join(args.output_folder, dataset_name)

DEBUG = args.debug

# Detector parameters
alpha = args.alpha
KNN = args.k

detector = AD(data_path, label_path, limit, train_size, KNN, alpha, DEBUG)
detector.initialize()

i = 0
while True:
    if i % 100 == 0:
        print('Processing batch {}'.format(i))
    res = detector.test_next_batch()
    if res == -1:
        detector.anomalyDetector.save_state(dataset_name, output_path)
        break
    i += 1
print('ODIT testing completed')

