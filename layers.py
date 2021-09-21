import numpy as np

#Just the layer every other layer inherits from.
class Layer():
    def __init__(self):
        self.input = None
        self.output = None
        self.l_rate = None

    def forward(self, input):
        pass

    def backprop(self, d_output, scaling_factor, regulizer):
        pass

#class Input_layer(Layer):
#    def __init__(self, input_size, output_size):
#        self.input_size = input_size
#        self.output_size = output_size
#
#    def forward(self, input):

#Just a function to initialize the weights. Want the initial weights to be small (between -0.1,0.1)
#Have not used glorot method.
def init_weights(min_value, max_value, output_size, input_size):
    weights = np.zeros((output_size, input_size), dtype=float)
    for i in range(output_size):
        for j in range(input_size):
            value = np.random.random()
            scaled_value = min_value + (value * (max_value - min_value))
            weights[i,j] = scaled_value
    return weights

#Dense layer aka the hidden layer for the most part. No activation function here.
class Dense_layer(Layer):
    def __init__(self, input_size, output_size, l_rate = 0.1):
        #self.weights = np.random.randn(output_size, input_size)
        self.weights = init_weights(-0.1, 0.1, output_size, input_size)
        #self.bias = np.random.randn(output_size, 1)
        self.bias = 0
        self.l_rate = l_rate

    #Method for summing the weight of the layer * the outputs from the previous layer.
    def forward(self, input):
        self.input = input
        summation = np.dot(self.weights, self.input) + self.bias
        return summation

    def backprop(self, d_output, scaling_factor, regulizer):
        d_weights = np.dot(d_output, self.input.T)
        d_bias = np.dot(d_output, 1) #gradient of inputs w.r.t biases are always 1 (thus dotted with 1)
        self.weights -= self.l_rate*d_weights #update weights
        self.addPenalty(regulizer, self.weights, scaling_factor) #Want to penalize high weights
        self.bias -= self.l_rate*d_bias
        d_input = np.dot(self.weights.T, d_output) #Gradient of loss wrt input. Used as input for backprop for the upstream layer.
        return d_input

    #Method for penalizing high weights (to avoid overfit/high variance)
    def addPenalty(self, regulizer, weights, scaling_factor):
        if (regulizer == 'L1'):
            self.weights -= (np.sign(weights) * scaling_factor)
        elif regulizer == 'L2':
            self.weights -= (weights * scaling_factor)

class Activation_layer(Layer):
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.l_rate = None

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backprop(self, d_output, scaling_factor = 0, regulizer = None):
        return np.multiply(d_output, self.activation_derivative(self.input))

class Sigmoid_layer(Activation_layer):
    def __init__(self):
        sigmoid = lambda x1 : np.exp(x1) / (1 + np.exp(x1))
        sigmoid_derivative = lambda x2 : sigmoid(x2)*(1 - sigmoid(x2))
        super().__init__(activation_derivative=sigmoid_derivative, activation=sigmoid)

class Tanh_layer(Activation_layer):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_prime)

class Relu_layer(Activation_layer):
    def __init__(self):
        relu = lambda x1 : np.max(0,x1)
        relu_prime = lambda x2 : 0 if x2 < 0 else 1
        super().__init__(activation_derivative=relu_prime, activation=relu)


class Softmax_layer(Layer): #No weights and biases, and layer is kinda 1-to-1, so no need for init.

    def __init__(self, input_size):
        self.input_size = input_size

    def forward(self, input):
        self.input = input
        self.output = np.exp(input) / np.sum(np.exp(input), axis=0)
        return self.output #Output from doing the softmax

    def backprop(self, d_output, scaling_factor = 0, regulizer = None):
        s = self.output.reshape(-1, 1) #make column vector
        softmax_derivative = np.diagflat(s) - np.dot(s, s.T)
        d_input = np.dot(softmax_derivative, d_output)
        return d_input

def main():
    dense = Dense_layer(3,4)
    act = Activation_layer(4,5)
    print(isinstance(act,Dense))

if __name__ == '__main__':
    main()





