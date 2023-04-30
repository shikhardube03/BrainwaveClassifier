import numpy as np

class KNNNeuralNetwork:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        self.weights = None

    def train(self, trainingSets, epochs=10, batch_size=32, learning_rate=0.001):
        X_train_all, X_test_all, y_train_all, y_test_all = [[],[],[],[]]
        for trainingData in trainingSets:
            channel_y_list = [np.stack(channel[0][1]) for channel in trainingData.values()]
            channel_data = np.stack(channel_y_list, axis=-1)
            channel_means = channel_data.mean(axis=1)
            X_train = channel_means
            y_train = trainingData[0][1]
            X_train_all.append(X_train)
            y_train_all.append(y_train)

        X_train_all = np.concatenate(X_train_all, axis=0)
        y_train_all = np.concatenate(y_train_all, axis=0)

        num_batches = len(X_train_all) // batch_size
        if self.weights is None:
            self._initialize_weights(X_train_all.shape[1])

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in range(num_batches):
                batch_X = X_train_all[batch*batch_size:(batch+1)*batch_size]
                batch_y = y_train_all[batch*batch_size:(batch+1)*batch_size]

                y_pred = self._forward_pass(batch_X)
                loss_value = np.mean((batch_y - y_pred)**2)
                epoch_loss += loss_value

                grad = self._backward_pass(batch_X, batch_y, y_pred)
                self._update_weights(grad, learning_rate)

            print(f"Epoch {epoch+1}: loss = {epoch_loss/num_batches:.4f}")

    def _initialize_weights(self, input_dim):
        self.weights = [
            np.random.randn(input_dim, 64) / np.sqrt(input_dim),
            np.zeros(64),
            np.random.randn(64, 10) / np.sqrt(64),
            np.zeros(10)
        ]

    def _forward_pass(self, X):
        layer1_weights = self.weights[0]
        layer1_bias = self.weights[1]
        layer2_weights = self.weights[2]
        layer2_bias = self.weights[3]
        layer1_output = np.dot(X, layer1_weights) + layer1_bias
        layer1_output = np.maximum(0, layer1_output)  # ReLU activation function

        layer2_output = np.dot(layer1_output, layer2_weights) + layer2_bias
        return layer2_output


    def _backward_pass(self, X, y, y_pred):
        layer1_weights = self.weights[0]
        layer2_weights = self.weights[2]

        # Gradient of loss w.r.t. output of layer 2
        grad_loss_output2 = y_pred - y
        grad_output2_weights = self.layer1_output.T

        # Gradient of loss w.r.t. output of layer 1
        grad_loss_output1 = np.dot(grad_loss_output2, layer2_weights.T)
        grad_output1_weights = X.T

        # Gradient of loss w.r.t. weights of layer 2
        grad_loss_weights2 = np.dot(self.layer1_output.T, grad_loss_output2)

        # Gradient of loss w.r.t. weights of layer 1
        grad_loss_weights1 = np.dot(X.T, grad_loss_output1)
       
