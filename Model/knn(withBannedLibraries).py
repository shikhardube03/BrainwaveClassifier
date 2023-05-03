import classify as classify
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import pickle 
class KNN(classify.Classifier):
    def __init__(self, n):
        self.model = KNeighborsClassifier(n_neighbors=n, weights='distance')

    def train(self, trainingSets):
        X_train_all, X_test_all, y_train_all, y_test_all = [[],[],[],[]]
        for trainingData in trainingSets:
          #  -> each group is 1 sec w 190 rows
            channel_y_list = [np.stack(channel[0][1]) for channel in trainingData.values()]
            channel_data = np.stack(channel_y_list, axis=-1)
            channel_means = channel_data.mean(axis=1)
            X_train = channel_means
            y_train = trainingData[0][1]
            X_train_all.append(X_train)
            y_train_all.append(y_train)

        X_train_all = np.concatenate(X_train_all)
        y_train_all = np.concatenate(y_train_all)
        self.model.fit(X_train_all, y_train_all)
    
    def saveModel(self, location):
        knnPickle = open(location, 'wb')
        pickle.dump(self.model, knnPickle) 

    def loadModel(self, location):
        self.trainedModel = pickle.load(open(location, 'rb'))

    def classify(self, observation):
        result = self.trainedModel.predict(observation)
        return result[0]
    