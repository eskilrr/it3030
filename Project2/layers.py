import numpy as np
from PIL import Image
import cv2

def processImage(image):
    image = cv2.imread(image)
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    return image

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

#Method for initializing the weights for the conv layer.
def init_weights_conv(num_channels, prev_num_channels, filter_size, min_value, max_value, dimension):
    n_h = filter_size if dimension == 2 else 1
    n_v = filter_size
    weights = np.zeros((num_channels, prev_num_channels, n_h, n_v))
    for c in range(num_channels):
        for h in range(n_h):
            for v in range(n_v):
                value = np.random.random()
                scaled_value = min_value + (value * (max_value - min_value))
                weights[c,:,h,v] = scaled_value
    return weights


class Conv_layer(Layer):
    def __init__(self, filter_size, dim, prev_num_channels, num_channels, padding = 0, stride = 1, l_rate = 0.1, learning = True):
        self.l_rate = l_rate
        self.filter_size_h = filter_size if dim == 2 else 1
        self.filter_size_v = filter_size

        self.padding = padding
        self.stride = stride
        self.channels = num_channels
        self.prev_num_channels = prev_num_channels
        self.weights = np.ones((num_channels, prev_num_channels, filter_size, filter_size))
        self.weights = init_weights_conv(num_channels=num_channels,prev_num_channels=prev_num_channels,
                                         filter_size=filter_size, dimension = dim, min_value=-0.1, max_value=0.1)

        self.learning = learning #Wheither the weights should be updated

    #Method for the forward pass.
    def forward(self, input):
        self.input = input

        #Get the dimensions of the output image, given the dimensions of the padded input + the stride of the filter.
        input_dim_C, input_dim_H, input_dim_V = input.shape
        dim_H = int((input_dim_H - self.filter_size_h + 2 * self.padding) / self.stride) + 1
        dim_V = int((input_dim_V - self.filter_size_v + 2 * self.padding) / self.stride) + 1

        #pad the input
        input_padded = self.pad_image(input, self.padding)
        output_image = np.zeros((self.channels, dim_H, dim_V)) #Init the output

        for channel in range(self.channels): #Loop over the output image
            for h in range(dim_H):
                for v in range(dim_V):
                    h_start = h*self.stride #positions for the filter at the current location
                    h_end = h_start + self.filter_size_h
                    v_start = v*self.stride
                    v_end = v_start + self.filter_size_v

                    input_part = input_padded[:, h_start:h_end, v_start:v_end] #slice the input to be filtered
                    output_image[channel, h, v] = self.apply_filter(input_part, channel = channel)  #apply the filter specified above
        self.output_image = output_image
        return output_image #return the filtered image to the next layer

    #Method for the backward-pass (learning-phase).
    def backprop(self, d_output, scaling_factor, regulizer):
        #Reshape the image to make it the same shape as the output of this layer.
        d_output = d_output.reshape(self.output_image.shape)

        (num_channels_prev, n_H_prev, n_V_prev) = self.input.shape
        (num_channels, n_H, n_V) = d_output.shape

        #Initialize the gradient of the loss wrt to both the input and the weights, respectively.
        da_prev = np.zeros((num_channels_prev, n_H_prev, n_V_prev))
        d_weights = np.zeros((num_channels, num_channels_prev, self.filter_size_h, self.filter_size_v))

        #Pad the input so dimensions are the same as in forward.
        a_prev_pad = self.pad_image(self.input, self.padding)
        da_prev_pad = self.pad_image(da_prev, self.padding)

        for c in range(self.channels): #Loop over thge gradient passed back from the previous layer (loss wrt output of this layer)
            for h in range(n_H):
                for v in range(n_V):
                    # positions for the filter at the current location
                    horiz_start = h*self.stride
                    horiz_end = horiz_start + self.filter_size_h
                    vert_start = v*self.stride
                    vert_end = vert_start + self.filter_size_v

                    a_slice = a_prev_pad[:, horiz_start:horiz_end, vert_start:vert_end] #Retrieve the region that contributes to the gradient wrt the input and the weight
                    #Compute the contributed gradients
                    da_prev_pad[:, horiz_start:horiz_end, vert_start:vert_end] += self.weights[c, :, :, :] * d_output[c, h, v]
                    d_weights[c, :, :, :] += a_slice * d_output[c, h, v]
        if self.learning: self.weights -= self.l_rate * d_weights #Wan't some option to check wheither updating the conv-weights
                                                           # actually contributes to the accuracy.
        if self.padding == 0:
            da_prev[:,:,:] = da_prev_pad
        else:
            da_prev[:, :, :] = da_prev_pad[:, self.padding:-self.padding, self.padding:-self.padding] #remove padding if it exists

        return da_prev #pass backward

    #Method for padding the image using numpys built in function.
    def pad_image(self, image, padding):
        padded = np.pad(image, ((padding,padding), (padding, padding),(0,0)), constant_values=0)
        return padded

    #Method for applying a filter over a given region (i.e input padded)
    def apply_filter(self, input_padded, channel):
        morn = np.multiply(input_padded,self.weights[channel,...])
        summed = np.sum(morn)
        return summed



#Dense layer aka the hidden layer for the most part. No activation function here.
class Dense_layer(Layer): #Note that the input size is set to none for the first dense after conv. Figured it was easier to just initialize the weights on the forward pass
                          # rather than before the training starts like in the previous project
    def __init__(self, input_size, output_size, l_rate = 0.1):
        #self.weights = np.random.randn(output_size, input_size)
        if input_size != None:
            self.weights = init_weights(-0.1, 0.1, output_size, input_size)
        self.input_size = input_size
        self.output_size = output_size
        self.bias = np.random.randn(output_size, 1)
        self.bias = 0
        self.l_rate = l_rate

    #Method for summing the weight of the layer * the outputs from the previous layer.
    def forward(self, input):
        if len(input.shape) > 2: # initializing the input size and the weights
            input_size = np.size(input)
            self.input = input.reshape(input_size, 1) #Reshaping the input from the conv to make a column vector, so all else become equal as in prev project
            if self.input_size == None:
                self.weights = init_weights(-0.5, 0.5, self.output_size, input_size)
                self.input_size = input_size
        else:
            self.input = input
        summation = np.dot(self.weights, self.input) + self.bias
        return summation

    #Backprop same as in previous project
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

#Layers for the activation functions. Relu worked best. Probably should have implemented Selu or leaky rely too, was short on time.
#using inheritance to reuse the forward and backward pass for the different activation functions.
class Activation_layer(Layer):
    def __init__(self, activation, activation_derivative, dense):
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.dense = dense
        self.l_rate = None

    def forward(self, input):
        self.input = input
        self.output = self.activation(self.input, dense=self.dense)
        return self.output

    def backprop(self, d_output, scaling_factor = 0, regulizer = None):
        d_output = d_output.reshape(self.output.shape)
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
    def __init__(self, dense):
        super().__init__(activation_derivative=self.relu_prime, activation=self.relu, dense=dense)

    def relu(self, input, dense):
        num_channels, n_h, n_v = input.shape
        zeros = np.zeros((num_channels,n_h,n_v))
        output = np.maximum(zeros,input)
        return output

    def relu_prime(self, input):
        num_channels, n_h, n_v = input.shape
        output = np.zeros((num_channels, n_h, n_v))
        for c in range(num_channels):
            for h in range(n_h):
                for v in range(n_v):
                    if input[c, h, v] < 0:
                        output[c, h, v] = 0
                    else:
                        output[c,h,v] = 1
        return output

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
    conv = Conv_layer(3, 1, 1)
    img = processImage('C:\\Users\eskil\PycharmProjects\it3030\Project2\kretskort.jpg')
    filtered = conv.forward(np.array([img]))
    filtered = filtered[0]
    cv2.imwrite('kretskort_filtered.jpg', filtered)

if __name__ == '__main__':
    main()

