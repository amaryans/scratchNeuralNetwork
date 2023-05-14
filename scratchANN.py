from cgi import FieldStorage
from queue import Full
from re import I
import numpy as np
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
    def activate(self, net):
        # can update to change the slope if needed, but right now y=x
        if self.activation == "linear":
            output = net
        elif self.activation == "logistic":
            output = 1 / (1 + np.exp(-net))
        return output

    # Calculate the output of the neuron. Should save the input and output for back-propagation.  
    # This is where we should calculate the net which will then be use above to calculate the activation 
    def calculate(self, input):
        # calculating a neurons output is as easy as multiplying the input value by the weights and adding the bias
        self.input = input
        if len(self.input) == self.inputNum:
            self.input = np.append(self.input, 1)
        net = np.sum(np.multiply(self.input, self.weights))
        self.output = self.activate(net)
        return self.output

    # This method returns the derivative of the activation function with respect to the net (do/dn)
    def activationDerivative(self):
        if self.activation == "linear":
            self.deriv = self.output
        elif self.activation == "logistic":
            self.deriv = self.output * (1 - self.output)
        return self.deriv
        
    # This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcPartialDerivative(self, deltaTimesW):
        # partial derivative for a neuron is the delataTimesW multiplied by the input
        # this is used in backpropogation to update the weights in order to continue learn
        self.deltaTimesW = np.multiply(deltaTimesW, self.activationDerivative())
        self.updateWeight()
        return self.deltaTimesW * self.weights

    # Simply update the weights using the partial derivatives and the leranring weight
    def updateWeight(self):
        print("weights:" + str(self.weights))
        print(str(np.multiply(self.deltaTimesW, self.input)))
        self.weights = self.weights - (self.lr * self.deltaTimesW * self.input)
        print("updated weights:" + str(self.weights))


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
        self.inputNum = inputNum
        # building a layer of neurons each with inputNum number of inputs
        for i in range(self.numOfNeurons):
            self.neurons.append(Neuron(activation, inputNum, lr, weights[i]))

        
    #calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)      
    def calculate(self, input):
        self.input = input
        output = []
        for neuron in self.neurons:
            neuron.calculate(input)
            output.append(neuron.output)
        return output
            
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its own w*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcWDeltas(self, wtimesdelta):
        nextWDeltas = np.zeros(shape=(self.numOfNeurons, self.inputNum + 1))
        for i in range(self.numOfNeurons):
            nextWDeltas[i] = self.neurons[i].calcPartialDerivative(wtimesdelta[i])
        return np.sum(nextWDeltas, axis = 0)
        
#An entire neural network        
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights (or else initialize randomly)
    def __init__(self,numOfLayers,numOfNeurons, inputSize, activation, lossFunction, lr, weights=[], epochs=1):
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
        self.epochs = epochs
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
        if self.lossFunction == "leastSquares":
            self.lossVal = 0.5 * np.sum(np.power((y-yp), 2))
        elif self.lossFunction == "meanSquares":
            self.lossVal = np.sum(np.power((y-yp), 2)) / len(yp)
    
    # Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)  
    # Lossderiv is part of delta      
    def lossDeriv(self,yp,y):
        self.lossDerivative = (yp-y)
        
    
    #Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values         
    def train(self,x,y):

        output = self.calculate(x)
        finalLayerOutput = output[-1]
        self.calculateLoss(finalLayerOutput, y)
        self.lossDeriv(finalLayerOutput, y)
        print("Final Layer Output")
        print(finalLayerOutput)
        nextWDeltas = self.lossDerivative
        print(nextWDeltas)
        for layer in reversed(self.layers):
            nextWDeltas = layer.calcWDeltas(nextWDeltas)
        

    def returnLoss(self):
        return self.lossDerivative


def plotTestResults(nnOuputs, legend, title):
    fig,ax = plt.subplots()
    for y, label in zip(nnOuputs, legend):
        x = np.linspace(1, len(y), len(y))
        ax.plot(x, y, label = label)
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel("epochs")
        ax.set_ylabel("value")

def plotLoss(loss, title):
    fig,ax = plt.subplots()
    lenLoss = len(loss)
    x = np.linspace(1, lenLoss, lenLoss)
    ax.plot(x, loss)
    ax.set_title(title)
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")

if __name__=="__main__":
    if len(sys.argv) < 2:
        # Starting point
        lr = 0.1
        weights = np.array([[[0.4, 0.45, 0.3]]])
        numOfLayers = 1
        numOfNeurons = np.array([1])
        numOfInputs = 2
        activation = "logistic"
        loss = "leastSquares"
        nn = NeuralNetwork(numOfLayers, numOfNeurons, numOfInputs, activation, loss, lr, weights, epochs = 100)
        highBit = 1
        lowBit = 0
        inputs = np.array([[lowBit, highBit], [highBit, lowBit], [lowBit,lowBit], [highBit,highBit]])
        y = np.array([[lowBit], [lowBit], [lowBit], [highBit]])
        numOfEpochs = 100
        lossValues = []
        y_0_0 = []
        y_0_1 = []
        y_1_0 = []
        y_1_1 = []
        for i in range(numOfEpochs):
            nn.train(inputs[1], y[1])
            y_1_0.append(nn.output[0])
            nn.train(inputs[0], y[0])
            y_0_1.append(nn.output[0])
            nn.train(inputs[3], y[3])
            y_1_1.append(nn.output[0])
            nn.train(inputs[2], y[2])
            y_0_0.append(nn.output[0])
            lossValues.append(nn.returnLoss())
        
        plotTestResults([y_0_0, y_0_1, y_1_0, y_1_1], ["0,0", "0,1", "1,0", "1,1"], "Binary AND Function\nlearning rate: %0.3f" % lr)
        plotLoss(lossValues, "Binary AND Function Loss\nlearning rate: %0.3f" % lr)
        plt.show()
        
    elif sys.argv[1] == "singleStep":
        # Run single step
        pass

    elif sys.argv[1] == "and":
        # Train to recognize "And" logic gate
        pass