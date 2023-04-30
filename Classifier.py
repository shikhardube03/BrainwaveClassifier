import numpy as np

class KNNNeuralNetwork:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        distances = np.sqrt(np.sum((X_test[:, np.newaxis, :] - self.X_train) ** 2, axis=2))
        nearest_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        nearest_labels = self.y_train[nearest_indices]
        return np.argmax(np.apply_along_axis(lambda x: np.bincount(x, minlength=len(np.unique(self.y_train))), axis=1, arr=nearest_labels), axis=1)


