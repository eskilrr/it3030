import numpy as np
from layers import *
#from data import *
from matplotlib import pyplot as plt



class neuralNetwork():

    def __init__(self):
        self.layers = []
        self.regulizer = 0
        self.loss_function = self.mse
        self.d_loss_function = self.mse_prime
        self.scaling_factor = 0

    def add_loss_function(self, loss_function):
        if (loss_function == "cross_entropy"):
            self.loss_function = self.cross_entropy
            self.d_loss_function = self.cross_entropy_deriv

    def add_regulizer(self,regulizer, scaling_factor):
        self.regulizer = regulizer
        self.scaling_factor = scaling_factor

    def cross_entropy(self, targets, predictions):
        return -1 * np.sum(targets * np.log(predictions))

    def cross_entropy_deriv(self, targets, predictions):
        return np.where(predictions != 0, -targets / predictions, 0.0)

    def mse(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    def mse_prime(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / np.size(y_true)

    def add_dense_layer(self, input_size, output_size):
        layer = Dense_layer(input_size, output_size)
        self.layers.append(layer)

    def add_activation_layer(self, activation_function):
        if (activation_function == "sigmoid"):
            layer = Sigmoid_layer()
            self.layers.append(layer)

        elif (activation_function == "tanh"):
            layer = Tanh_layer()
            self.layers.append(layer)

    def add_softmax_layer(self, input_size):
        layer = Softmax_layer(input_size)
        self.layers.append(layer)

    def train_model(self, x_train, y_train, x_valid, y_valid, epochs, verbose = False):
        step_size = 20 # How frequent the loss should be plotted/printed
        training_losses = [] #record for plot-purposes.
        training_loss = 0

        validation_losses = [] #record for plot-purposes
        validation_accuracies = [] #record for plot-purposes
        for e in range(epochs):
            correct = 0 #n correct predictions
            n = 1
            for x, y in zip(x_train, y_train):
                # forward
                output = x
                for layer in self.layers: #forward-prop through all layers
                    output = layer.forward(output)

                predicted = np.argmax(output)
                true = np.argmax(y)
                if (predicted == true): #check if predicted is correct
                    correct += 1

                # calculate loss
                training_loss += self.loss_function(y, output)
                mod = n % step_size
                n += 1
                if mod == 0:
                    training_loss /= step_size
                    if verbose: print('%d/%d %d/%d, loss=%f' % (n, len(x),  e + 1, epochs, training_loss))
                    training_losses.append(training_loss) #record the loss

                # backward
                grad = self.d_loss_function(y, output) #first jacobi: Gradient wrt output
                for layer in reversed(self.layers): #backprop through all layers
                    grad = layer.backprop(grad, self.regulizer, self.scaling_factor)

            training_accuracy = (correct / len(x_train)) * 100

            print("Training-accuracy: ", training_accuracy, " %")
            validation_accuracy, validation_loss = self.make_prediction(x_valid, y_valid)
            validation_losses.append(validation_loss), validation_accuracies.append(validation_accuracy)
            print("\n")

        #Code for plotting loss for train and validation + accuracy for validation.
        x_axis2 = np.linspace(0, step_size*len(training_losses), len(training_losses))
        plt.plot(x_axis2, training_losses, label = "training")
        x_axis1 = np.linspace(0, epochs*len(x_train), epochs)
        plt.plot(x_axis1, validation_losses, label = "validation")
        plt.xlabel("training examples")
        plt.ylabel("Loss")
        plt.show()

        plt.plot(validation_accuracies)
        plt.xlabel("training examples")
        plt.ylabel("accuracy")
        plt.show()

    #Method for validation/test-set
    def make_prediction(self, x_test, y_test):
        correct = 0
        loss_sum = 0
        for x, y in zip(x_test, y_test):
            output = x
            for layer in self.layers:
                output = layer.forward(output)
            predicted = np.argmax(output)
            true = np.argmax(y)
            if (predicted == true):
                correct += 1
            loss_sum += self.loss_function(y, output)
        accuracy = (correct / len(x_test)) * 100
        loss = loss_sum / len(x_test)
        print("valid/test accuracy: ", accuracy, " %")
        return accuracy, loss
