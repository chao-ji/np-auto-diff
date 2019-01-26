# Implementing Automatic Differentiation in python

* As of now this project is only intended for illustrative purposes and not for production.
* Implemented in pure python, the only exception being that `numpy` is used for computational heavy-lifting tasks such as matrix multiplication.
* Neural networks are implemented as computational graphs by wiring up `Operation`-like objects (like in `tensorflow`), such as 2D convolution, pooling, etc.
* See [this](https://github.com/chao-ji/reverse-mode-auto-differentiation/blob/master/convnet_mnist_demo.py) for a simple demo of a convolutional network on MNIST dataset. This ConvNet contains two convolutional layers each followed by a max-pooling layer, then a fully connected layer, and also a dropout layer right before the final readout layer (as in this [tensorflow demo](https://www.tensorflow.org/get_started/mnist/pros)). Gradient is updated using ADAM algorithm. Should take about 20 min to run on a Duo-core processor and the accuracy on the test test should be around 0.97 with only 500 training iterations.
