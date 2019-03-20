# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 06:29:37 2019

@author: Shrey
"""

# Import our dependencies
# For custom implementation
import numpy as np
from utilities import data_util as du


# Create our Artificial Neural Network class
class ArtificialNeuralNetwork:

    # initializing the class
    def __init__(self):

        # generating the same synaptic weights every time the program runs
        np.random.seed(1)

        # synaptic weights (3 × 4 Matrix) of the hidden layer
        self.w_ij = 2 * np.random.rand(3, 4) - 1

        # synaptic weights (4 × 1 Matrix) of the output layer
        self.w_jk = 2 * np.random.rand(5, 1) - 1

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

    def train(self, X, Y, learning_rate=0.5, iterations=1):

        # x: training set of data
        # y: the actual output of the training data

        for i in range(iterations):
            num_examples = 7
            i = np.random.randint(len(X) - num_examples)
            x = X[i:i + num_examples, :]

            # ones_array = np.atleast_2d(np.ones(x.shape[0]))
            # x = np.concatenate((x, ones_array.T), axis=1)

            x = du.append_ones(x)

            y = Y[i:i + num_examples, :]

            z_ij = np.dot(x, self.w_ij)  # the np.dot product of the weights of the hidden layer and the inputs
            a_ij = self.sigmoid(z_ij)  # applying the Sigmoid activation function

            a_ij = du.append_ones(a_ij)
            # np.concatenate((a_ij, ones_array.T), axis=1)

            z_jk = np.dot(a_ij, self.w_jk)  # the same previous process will be applied to find the predicted output
            a_jk = self.sigmoid(z_jk)

            #            print("a_jk: {0} \n y: {1} and matrix div: {2}".format(a_jk, y, y/a_jk))
            dl_jk = -y / a_jk + (1 - y) / (1 - a_jk)  # the derivative of the cross entropy loss wrt output
            da_jk = self.sigmoid_derivative(
                a_jk)  # the derivative of Sigmoid  wrt the input (before activ.) of the output layer
            dz_jk = a_ij  # the derivative of the inputs of the hidden layer (before activation) wrt weights of the output layer

            dl_ij = np.dot(da_jk * dl_jk,
                        self.w_jk.T)  # the derivative of cross entropy loss wrt hidden layer input (after activ.)
            da_ij = self.sigmoid_derivative(
                a_ij)  # the derivative of Sigmoid wrt the inputs of the hidden layer (before activ.)
            dz_ij = x  # the derivative of the inputs of the hidden layer (before activation) wrt weights of the hidden layer

            # calculating the gradient using the chain rule
            gradient_ij = np.dot(dz_ij.T, dl_ij * da_ij)
            gradient_ij = np.delete(gradient_ij, -1, 1)
            gradient_jk = np.dot(dz_jk.T, dl_jk * da_jk)

            #            print("Shapes:")
            #            print("dz_ij: {0}\nda_ij: {1}\ndl_ij{2}".format(dz_ij.shape, da_ij.shape, dl_ij.shape))
            #            print("dz_jk: {0}\nda_jk: {1}\ndl_jk{2}".format(dz_jk.shape, da_jk.shape, dl_jk.shape))

            # calculating the new optimal weights
            self.w_ij = self.w_ij - learning_rate * gradient_ij
            self.w_jk = self.w_jk - learning_rate * gradient_jk

            # printing the loss of our neural network after each 1000 iteration
            if i % 1000 == 0 in range(iterations):
                print("loss: ", self.crossentropyerror(a_jk, y))

    def predict(self, inputs):

        inputs = du.append_ones(inputs)

        # predicting the class of the input data after weights optimization

        output_from_layer1 = self.sigmoid(np.dot(inputs, self.w_ij))  # the output of the hidden layer
        # output_from_layer1 = du.append_ones(output_from_layer1)
        #        output_from_layer1 = np.append(output_from_layer1, [[1]], axis = 1)

        output_from_layer1 = du.append_ones(output_from_layer1)
        output_from_layer2 = self.sigmoid(np.dot(output_from_layer1, self.w_jk))  # the output of the output layer

        return output_from_layer1, output_from_layer2

    # the function will print the initial starting weights before training
    def synaptic_weights(self):

        print("Layer 1 (3 neurons, each with 3 inputs except the bias neuron): ")

        print("w_ij: ", self.w_ij)

        print("Layer 2 (1 neuron, with 4 inputs): ")

        print("w_jk: ", self.w_jk)

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
        return correct_predictions / total_instances

        print("Incorrect predictions: ", incorrect)
        #
        # # for instance in x:
        # (_, out) = self.predict(x)
        #
        # for i, [prediction] in enumerate(out):
        #     if round(prediction) == 0:
        #         predicted_label = 0
        #     else:
        #         predicted_label = 1
        #
        #     if y[i] == predicted_label:
        #         correct_predictions += 1
        #     else:
        #         incorrect += [(i, prediction, y[i][0])]

        print("Correct predictions: ", correct_predictions)
        print("Total test instances: ", total_instances)


def ann_main(x, y, x_test, y_test, epochs = 100):
    ANN = ArtificialNeuralNetwork()

    ANN.synaptic_weights()

    # the training inputs
    # the last column is used to add non linearity to the classification task

    # the training outputs
    learning_rate = 1


    ANN.train(x, y, learning_rate, epochs)

    # Printing the new synaptic weights after training
    print("New synaptic weights after training: ")
    print("w_ij: ", ANN.w_ij)
    print("w_jk: ", ANN.w_jk)

    # Our prediction after feeding the ANN with new set of data
    # random_test = np.array([[-5.9, -0.006, 1], [-5.9, 0.006, 1], [5.9, -0.006, 1], [5.9, 0.006, 1], [-0.0059, -6, 1], [-0.0059, 6, 1],[0.0059, -6, 1], [0.0059, 6, 1]])
    (_, out_layer_2) = ANN.predict(x_test)

    print(ANN.evaluate(out_layer_2, y_test))

    # print(ANN.evaluate(x_test, y_test))


if __name__ == "__main__":
    max_range = 10

    (x, y) = du.load_data(max_range)
    (x, y) = du.shuffle_data(x, y)

    (x_test, y_test) = du.load_data(max_range, single_class_side_number=24)
    (x_test, y_test) = du.shuffle_data(x_test, y_test)

    ann_main(x, y, x_test, y_test)
