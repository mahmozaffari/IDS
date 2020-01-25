import os
import csv
import numpy as np


class DL:
    def __init__(self, data_file, label_file, batch_size=1000, debug=False):
        if debug:
            print('[Info] Initiating the DataLoader object...')
        self.data_file = data_file
        self.label_file = label_file
        self.batch_size = batch_size
        self.row_index = 0    # index of next row to load
        self.DEBUG = debug
        self.check_file()
        self.eof = False

    def check_file(self):
        dfile_exist = os.path.exists(self.data_file)
        lfile_exist = os.path.exists(self.label_file)
        if not dfile_exist or not lfile_exist:
            print('[Error] The file(s) doesn\'t exist.')
            raise Exception()

        self.dfh = open(self.data_file, 'r')
        self.lfh = open(self.label_file, 'r')
        self.datain = csv.reader(self.dfh)
        self.labelin = csv.reader(self.lfh)
        self.dim = len(self.datain.__next__()) - 1
        self.labelin.__next__()
        #if self.DEBUG:
        #    print('[Info] Data dimensionality is: {}'.format(self.dim))
        self.row_index += 1

    def __call__(self, batch=None):
        # load the next batch
        if self.eof:
            raise Exception('File End reached.')
        if batch is None:
            batch = self.batch_size
        if self.DEBUG:
            print('[Info] Fetching {} data...'.format(batch))
        # retrieve next batch of size specified
        batch_data = np.empty((batch, self.dim), dtype=float)
        batch_label = np.empty(batch, dtype=int)
        for i in range(batch):  #first column is
            try:
                d = self.datain.__next__()
                l = self.labelin.__next__()
                batch_data[i, :] = d[-115:]
                batch_label[i] = l[-1]
            except StopIteration:
                print('[Info] End of file!')
                batch_data = batch_data[0:i]
                batch_label = batch_label[0:i]
                self.eof = True
                break

        if self.DEBUG:
            print('[Info] {} data instances retrieved.'.format(batch_data.shape))
        return batch_data, batch_label

    def fetch_train_data(self, batch=None):
        train_data, train_label = self.__call__(batch)
        idx = np.where(train_label == 1)
        if len(idx[0]) > 0:
            print('[Info]: Training data contains anomaly.')
            train_data = np.delete(train_data, idx, axis=0)
            train_label = np.delete(train_label, idx, axis=0)
            print('[Info]: Training data reduced by {}'.format(len(idx[0])))
        return train_data

    def terminate(self):
        self.data_file_h.close()
        self.label_file_h.close()
