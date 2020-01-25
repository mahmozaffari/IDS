import numpy as np
import pickle
import os

class ODIT:
    def __init__(self, train_data, k=1, alpha=0.05, mode='both', debug=False):
        # k : number of nearest neighbors to compute
        # alpha : significance level
        # mode: 'distance', 'percentile', 'both'
        self.DEBUG = debug
        self.k = k
        self.alpha = alpha
        self.mode = mode
        self.nominal_data = train_data
        self.dim = self.nominal_data.shape[1]
        # processed
        self.norm = {'min': np.min(self.nominal_data,axis=0), 'max': np.max(self.nominal_data,axis=0)}
        self.evidences = {}
        self.evidences['percentile'] = np.empty(0, dtype=float)
        self.evidences['distance'] = np.empty(0, dtype=float)
        self.labels = np.empty(0, dtype=int)
        #   stats
        self.processed = 0
        self.counter = 0

    def normalize(self, data):
        return np.divide(data-self.norm['min'], self.norm['max']-self.norm['min'])

    def set_train_data(self, train_set):
        self.nominal_data = train_set
        self.dim = train_set.shape[1]

    def train(self):
        self.nominal_data = self.normalize(self.nominal_data)
        n = self.nominal_data.shape[0]
        self.D = np.zeros(n, dtype=float)
        for i, x in enumerate(self.nominal_data):
            R = np.repeat(x[np.newaxis, :], n, axis=0)
            distances = np.sqrt(np.sum((R - self.nominal_data)**2, axis=1))
            distances = np.sort(distances)
            assert(distances[0] == 0)
            self.D[i] = sum(distances[0:self.k+1])
        self.D = np.sort(self.D)
        if self.mode == 'both' or self.mode == 'distance':
            self.boundary_d = self.D[int(n*(1-self.alpha))]
            if self.DEBUG:
                print('[Info] Boundary distance is {}.'.format(self.boundary_d))

    def test(self, test_data, labels):
        self.counter += 1
        print('[Info] Processing batch {}'.format(self.counter))
        n = test_data.shape[0] # number of rows of data
        test_data = self.normalize(test_data)
        #p_evidence = np.zeros(n, dtype=float)
        #d_evidence = np.zeros(n, dtype=float)
        knn_distances = np.zeros(n, dtype=float)
        for i, x in enumerate(test_data):
            distances = np.sqrt(np.sum((x - self.nominal_data)**2,axis=1))
            distances = np.sort(distances)
            knn_distances[i] = np.sum(distances[0:self.k])
        p_evidence = self.compute_percentile_ev(knn_distances)
        d_evidence = self.compute_dist_ev(knn_distances)
        self.evidences['percentile'] = np.append(self.evidences['percentile'], p_evidence)
        self.evidences['distance'] = np.append(self.evidences['distance'], d_evidence)
        self.labels = np.append(self.labels, labels)
        self.processed += n
        return True

    def compute_dist_ev(self, d):
        return np.log(d) - np.log(self.boundary_d)

    def compute_percentile_ev(self, d):
        pt = np.empty_like(d)
        for i in range(len(d)):
            pt[i] = max(np.sum(self.D > d[i]) / len(self.D), 1e-6)
        return np.log(self.alpha / pt)

    def save_state(self, dataset_name, file_path):
        state = {'dataset': dataset_name, 'evidences': self.evidences, 'labels': self.labels, 'processed': self.processed,
                 'train_data': self.nominal_data, 'D': self.D, 'alpha': self.alpha, 'k': self.k }
        pickle_path = os.path.join(file_path,dataset_name)
        with open(file_path, 'wb') as f:
            pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)
            print('[Info] ODIT state saved')





