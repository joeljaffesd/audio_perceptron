'''

Joel A. Jaffe 2025-03-16

Let's start from first principles.

A perceptron that senses how light or dark a scene is,
and outputs a continuos value to determine whether it should move up or down.

This mimics the first "eye" that developed in early ocean life,
which could sense light and dark and move up or down to get balance exposure
to UV light.

The input layer is vector of values, which we may interpret as a rasterized image.

The hidden layer is a vector of values, like neurons.

The output layer is a single value, which we may interpret as a direction to move.
We could interpret as a force and integrate... or just as a position. 

Ok, so where's the audio part?
Well, maybe we could train this system to behave like an optical compressor.

The input layer is some picture or video of what a photoresistor sees, 
the output layer is the current output sample. 

But what will we train it on?
We need to record a vactrol's response to an image signal,
and then train the network to predict the output.

To learn a more general input/output mapping, 
the input layer can just be some signal history,
and the output once again, a single value.

'''

import numpy as np  

input_layer = []
hidden_layer = []
output_layer = 0

def sigmoid(x):
  return 1/(1+np.exp(-x))