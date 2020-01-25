import numpy as np
from DataLoader import DL
from ODIT import ODIT
import sys
import math


class AD:
    def __init__(self, data_path, label_path, limit, train_n=50000, k=1, alpha=0.05, debug=False):
        self.train_size = train_n
        self.DEBUG = debug
        self.limit = limit
        self.data_loader = DL(data_path, label_path, debug=debug)
        #train_data = self.data_loader.fetch_train_data(batch=train_n)

    def initialize(self):
        try:
            train_data = self.data_loader.fetch_train_data(batch=self.train_size)
            self.anomalyDetector = ODIT(train_data, debug=True)
            self.anomalyDetector.train()
            print('Boundary distances is {}'.format(self.anomalyDetector.boundary_d))
        except Exception as e:
            print('Initialization Failed')
            print(e)
            sys.exit(1)

    def test_next_batch(self):
        if self.anomalyDetector.processed >= self.limit:
            print('exceeded limit')
            return -1   # exceeded the test limit
        try:
            (test_set, test_label) = self.data_loader(batch=1000)
        except Exception as e:
            print(e)
            return -1   # no data left
        return self.anomalyDetector.test(test_set, test_label)


if __name__ == '__main__':
    ad = AD('.\\Data\\Mirai_dataset.csv', '.\\Data\\ARP MitM_labels.csv', debug=True)