from cgi import FieldStorage
from queue import Full
from re import I
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import sys
import math

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator as ML
from matplotlib.ticker import ScalarFormatter as SF
from matplotlib import ticker

class Neuron:
    #initilize neuron with activation type, number of inputs, learning rate, and possibly with set weights
    def __init__(self, activation, inputNum, lr, weights=None):
        self.activation = activation
        self.inputNum = inputNum
        self.lr = lr
        self.deltaTimesW = []
        self.weights = weights
        self.lenWeights = len(weights)
    
    # This method returns the activation of the net (NET IS WEIGHT * INPUT PLUS BIAS)
    def activate(self):
        # can update to change the slope if needed, but right now y=x
        if self.activation == "linear":
            self.output = self.net
        elif self.activation == "logistic":
            self.output = 1 / (1 + np.exp(-1 * self.net))
        return self.output

    # Calculate the output of the neuron. Should save the input and output for back-propagation.  
    # This is where we should calculate the net which will then be use above to calculate the activation 
    def calculate(self, input):
        # calculating a neurons output is as easy as multiplying the input value by the weights and adding the bias
        self.input = input
        self.net = np.sum(np.multiply(self.input, self.weights[0:self.inputNum])) + self.weights[self.inputNum]
        return self.net

    # This method returns the derivative of the activation function with respect to the net 
    def activationDerivative(self):
        if self.activation == "linear":
            self.deriv = self.output
        elif self.activation == "logistic":
            self.deriv = self.output * (1 - self.output)
        return self.deriv
        
    # This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcPartialDerivative(self, deltaTimesW):
        # partial derivative for a neuron is the delataTimesW multiplied by the input
        # this is used in backpropogation to update the weights in order to continue learning
        self.deltaTimesW = np.multiply(self.input, deltaTimesW)
        #self.deltaTimesW[0] = deltaTimesW
        return self.deltaTimesW

    # Simply update the weights using the partial derivatives and the leranring weight
    def updateWeight(self):
        self.weights = self.weights - lr * self.deltaTimesW


#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the input size, the leraning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self,numOfNeurons, activation, inputNum, lr, weights=None):
        """
        numOfNeurons: int
            Number of neurons in the layer
        activation: string
            linear - linear activation function
        inputNum: int
            number of inputs to each neuron
        lr: float
            learning rate of the neural network, usually on scale of 10e-2
        weights: 2D numpy array
            weights for each of the neurons
        """
        self.neurons = []
        self.numOfNeurons = numOfNeurons
        self.weights = weights
        # building a layer of neurons each with inputNum number of inputs
        for i in range(self.numOfNeurons):
            self.neurons.append(Neuron(activation, inputNum, lr, weights[i]))

        
    #calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)      
    def calculate(self, input):
        self.input = input
        output = []
        for neuron in self.neurons:
            neuron.calculate(input)
            neuron.activate()
            output.append(neuron.output)
        return output
            
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its own w*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcWDeltas(self, wtimesdelta):
        for i in range(self.numOfNeurons):
            print(self.input[i] * self.neurons[i].activationDerivative() * wtimesdelta[i])
        
#An entire neural network        
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights (or else initialize randomly)
    def __init__(self,numOfLayers,numOfNeurons, inputSize, activation, lossFunction, lr, weights=[]):
        """
        numOfLayers: int
            Number of layers in the Neural Network
        numOfNeurons: vector
            Number of neurons in the layer (e.g. [2,1] would have 2 neurons in the first layer and 1 in the second)
        activation: string
            linear - linear activation function
        inputSize: int
            number of inputs to each neuron
        loss: string
            loss function used for this network
        lr: float
            learning rate of the neural network, usually on scale of 10e-2
        weights: 3D numpy array
            weights for each of the neurons in each of the layers (e.g. following example above [[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], [[0.5, 0.5, 0.5]]], each layer, then each neuron)
        """
        self.layers = []
        self.lossFunction = lossFunction
        for i in range(numOfLayers):
            self.layers.append(FullyConnected(numOfNeurons[i], activation, inputSize, lr, weights[i]))
        pass
    
    def assignRandomWeights(self,numOlayers,numOneurons,inputsize):
        pass

    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,input):
        self.output = []
        for layer in self.layers:
            # input to the next layer is the output from the previous layer
            input = layer.calculate(input)
            self.output.append(input)
        return self.output
        
    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateLoss(self,yp,y):
        print(yp, y)
        if self.lossFunction == "leastSquares":
            self.lossVal = 0.5 * np.sum(np.power((y-yp), 2))
    
    # Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)  
    # Lossderiv is part of delta      
    def lossDeriv(self,yp,y):
        self.lossDerivative = -(y-yp)
        
    
    #Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values         
    def train(self,x,y):
        output = self.calculate(x)
        finalLayerOutput = output[-1]
        self.calculateLoss(finalLayerOutput, y)
        self.lossDeriv(finalLayerOutput, y)
        nextWDeltas = self.lossDerivative
        print(nextWDeltas)
        for layer in range(numOfLayers):
            nextWDeltas = self.layers[numOfLayers-layer-1].calcWDeltas(nextWDeltas)

    def returnLoss(self):
        return self.lossDerivative
    

if __name__=="__main__":
    if len(sys.argv) < 2:
        # Starting point
        lr = 0.1
        weights = np.array([[[0.15, 0.2, 0.35],[0.25, 0.30, 0.35]],
                            [[0.40, 0.45, 0.6],[0.50, 0.55, 0.6]]])
        numOfLayers = 2
        numOfNeurons = np.array([2, 2])
        numOfInputs = 2
        activation = "logistic"
        loss = "leastSquares"
        nn = NeuralNetwork(numOfLayers, numOfNeurons, numOfInputs, activation, loss, lr, weights)
        inputs = np.array([.05, .1])
        y = np.array([0.01,0.99])
        nn.train(inputs, y)
    
        
    elif sys.argv[1] == "singleStep":
        # Run single step
        pass

    elif sys.argv[1] == "and":
        # Train to recognize "And" logic gate
        pass