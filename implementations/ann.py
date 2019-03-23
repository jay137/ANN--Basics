# Import our dependencies
# For custom implementation
import numpy as np
from utilities import data_util as du


# Create our Artificial Neural Network class
class ArtificialNeuralNetwork:

    # initializing the class
    def __init__(self, layers = [2,2,2,1]):

        # generating the same synaptic weights every time the program runs
        np.random.seed(1)
        self.w = []
        for layer_index in range(1, len(layers)):
            self.w += [2 * np.random.rand(layers[layer_index - 1] + 1, layers[layer_index]) - 1]

        print("--Weights initialized--")

    def sigmoid(self, x):

        # The Sigmoid activation function will turn every input value into probabilities between 0 and 1
        # the probabilistic values help us assert which class x belongs to

        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):

        # The derivative of Sigmoid will be used to calculate the gradient during the backpropagation process
        # and help optimize the np.random starting synaptic weights

        return x * (1 - x)

    def crossentropyerror(self, a, y):

        # The cross entropy loss function
        # we use it to evaluate the performance of our model

        return - sum(y * np.log10(a) + (1 - y) * np.log10(1 - a))

    def forward_iteration(self, x, layers):
        a = [x]
        z = []

        for j in range(1, len(layers)):
            z += [np.dot(a[j - 1], self.w[j - 1])]
            activation = self.sigmoid(z[-1])
            if j != len(layers) - 1:
                a += [du.append_ones(activation)]
            else:
                a += [activation]

        return a,z

    def train(self, X, Y, layers=[2,2,2,1], learning_rate=0.5, iterations=1):

        # x: training set of data
        # y: the actual output of the training data

        for i in range(iterations):

            num_examples = 7
            i = np.random.randint(len(X) - num_examples)
            x = X[i:i + num_examples, :]
            x = du.append_ones(x)
            y = Y[i:i + num_examples, :]

            # Data check breakpoint
            (a,z) = self.forward_iteration(x, layers)

            # Activations and linear computation breakpoint
            dl = []
            da = []
            dz = []
            for j in range(len(layers) - 1, 0, -1):
                if j == (len(layers) - 1):
                    dl = [-y / a[j] + (1 - y) / (1 - a[j])] + dl
                else:
                    da_dl = da[0] * dl[0]
                    if da[0].shape[1] != layers[j+1]:
                        da_dl = np.delete(da_dl, -1, 1)
                    dl = [np.dot(da_dl, self.w[j].T)] + dl

                da = [self.sigmoid_derivative(a[j])] + da
                dz = [a[j-1]] + dz

            # deltas check breakpoint
            gradients  = []
            for j in range(1, len(layers)):
                da_dl = dl[j - 1] * da[j - 1]
                if da[j-1].shape[1] != layers[j]:
                    da_dl = np.delete(da_dl, -1, 1)
                gradients = gradients + [np.dot(dz[j-1].T, da_dl)]

            for j in range(len(self.w)):
                self.w[j] -= learning_rate * gradients[j]

            # printing the loss of our neural network after each 1000 iteration
            if i % 1000 == 0 in range(iterations):
                print("loss: ", self.crossentropyerror(a[-1], y))

    def synaptic_weights(self):
        for i in range(len(self.w)):
            print("--Layer {0}--".format(i+1))
            print(self.w[i])

    def predict(self, inputs, layers=[2,2,2,1]):

        inputs = du.append_ones(inputs)

        (a, _) = self.forward_iteration(inputs, layers)

        return a[-1]

    def evaluate(self, yhat, y):
        # x = du.append_ones(x)
        correct_predictions = 0
        total_instances = y.shape[0]
        incorrect = []

        for index, prediction in enumerate(yhat):
            if round(prediction[0]) == y[index][0]:
                correct_predictions += 1
            else:
                incorrect += [(prediction[0], y[index][0])]

        print("Incorrect predictions: ", incorrect)
        print("Correct predictions: ", correct_predictions)
        print("Total test instances: ", total_instances)
        return correct_predictions / total_instances


def ann_main(x, y, x_test, y_test, learning_rate = 1, epochs = 100):
    ANN = ArtificialNeuralNetwork()

    ANN.synaptic_weights()

    # the training outputs
    ANN.train(x, y, learning_rate = learning_rate, iterations = epochs)

    # Printing the new synaptic weights after training
    print("New synaptic weights after training: ")
    ANN.synaptic_weights()

    # Our prediction after feeding the ANN with new set of data
    output = ANN.predict(x_test)

    print(ANN.evaluate(output, y_test))