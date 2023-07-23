#!/usr/bin/env python

import numpy as np
import random

from softmax import softmax
from sigmoid import sigmoid, sigmoid_grad
from gradcheck import gradcheck_naive


def forward(data, label, params, dimensions):
    """
    runs a forward pass and returns the probability of the correct word for eval.
    label here is an integer for the index of the label.
    This function is used for model evaluation.
    """
    # Unpack network parameters
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Compute the probability
    # Forward propagation
    hidden_layer = sigmoid(np.dot(data, W1) + b1)
    output_layer = softmax(np.dot(hidden_layer, W2) + b2)

    # Extract the probabilities of the correct labels
    correct_label_probabilities = output_layer[np.arange(output_layer.shape[0]), label]
    return correct_label_probabilities


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    # Unpack network parameters
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Forward propagation
    layer1_activations = np.dot(data, W1) + b1
    layer1_outputs = sigmoid(layer1_activations)

    layer2_activations = np.dot(layer1_outputs, W2) + b2
    predictions = softmax(layer2_activations)

    # Backward propagation
    error = predictions - labels

    gradW2 = np.dot(layer1_outputs.T, error)
    gradb2 = np.sum(error, axis=0)

    backprop_grad = np.dot(error, W2.T) * sigmoid_grad(layer1_outputs)

    gradW1 = np.dot(data.T, backprop_grad)
    gradb1 = np.sum(backprop_grad, axis=0)

    # Stack gradients
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), gradW2.flatten(), gradb2.flatten()))

    # Calculate the cross-entropy cost
    cost = -np.sum(np.log(predictions[labels == 1]))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])  # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0, dimensions[2] - 1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
            dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
                    forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q1c_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")

    print("Your backward sanity checks passed!")


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
