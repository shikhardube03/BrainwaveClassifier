import numpy as np
import pickle

class KNN:
    def __init__(self, n):
        self.model = None
        self.n_neighbors = n

    def train(self, trainingSets):
        X_train_all, y_train_all = [], []
        for trainingData in trainingSets:
            channel_y_list = [np.stack(channel[0][1]) for channel in trainingData.values()]
            channel_data = np.stack(channel_y_list, axis=-1)
            channel_means = channel_data.mean(axis=1)
            X_train = channel_means
            y_train = trainingData[0][1]
            X_train_all.append(X_train)
            y_train_all.append(y_train)

        X_train_all = np.concatenate(X_train_all)
        y_train_all = np.concatenate(y_train_all)
        self.model = self._fit_model(X_train_all, y_train_all)

    def _fit_model(self, X_train, y_train):
        distances = np.sum(X_train**2, axis=1, keepdims=True) + np.sum(self.model**2, axis=1) - 2*np.dot(X_train, self.model.T)
        indices = np.argpartition(distances, self.n_neighbors, axis=0)[:self.n_neighbors,:]
        knn_labels = y_train[indices]
        weights = 1 / (distances[indices] + 1e-8)
        knn_weights = weights / np.sum(weights, axis=0)
        return knn_weights.dot(knn_labels.T).T

    def saveModel(self, location):
        with open(location, 'wb') as f:
            pickle.dump(self.model, f)

    def loadModel(self, location):
        with open(location, 'rb') as f:
            self.model = pickle.load(f)

    def classify(self, observation):
        distances = np.sum(observation**2, axis=1, keepdims=True) + np.sum(self.model**2, axis=1) - 2*np.dot(observation, self.model.T)
        indices = np.argpartition(distances, self.n_neighbors, axis=1)[:,:self.n_neighbors]
        knn_labels = self.model[indices]
        weights = 1 / (distances[:,indices] + 1e-8)
        knn_weights = weights / np.sum(weights, axis=1, keepdims=True)
        return np.argmax(np.sum(knn_weights*knn_labels, axis=1), axis=0)
