# Pbject Oriented Artificial Neural Netwokr
An very simple object oriented approach to creating an artifical neural network. 
While not the most efficient way of creating and training neural networks, this is a good way to understand the mathematics behind the "black box" of TensorFlow and other libraries

Classes:
- Neuron
  - Methods to calculate inputs, outputs, update weights, and activation function.
- FullyConnectedLayer
  - Essentially an array of Neurons which passes information between the layers
- NeuralNetwork
  - Creates the FullyConnectedLayers, and trains based on defined parameters

Future Development:
- Convolutional layers
- Max Pooling
- Flattening
- Dropout
- Visuals of training data for test cases
