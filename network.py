#this py file contains a generic class for the network

import numpy as np
import pandas as pd
from collections import defaultdict
from my_utils import act_softmax, act_relu, loss_fnc

class network_model:
    def __init__(self, no_hid_layer, X_train, Y_train, nueron_num_arr) -> None:
        self.hidden_layers = no_hid_layer
        self.input_size = X_train.shape[1]
        self.data = X_train
        self.label = Y_train
        self.lay_par = defaultdict(None)
        self.loss_by_epochs = []
        self.nue_nums = nueron_num_arr

    def layerInitialising(self, neuron_number):
        np.random.seed(0)
        # initailising all hidden layers, and also the output layer
        for i in range(self.hidden_layers + 1):
            self.weights = 0.1 * np.random.randn(self.input_size, neuron_number[i]) #initaialising weights to be within -1 to 1
            self.biases = np.zeros((1, neuron_number[i]))
            self.input_size = neuron_number[i]

            self.lay_par[f"lay${i+1}_w"] = self.weights
            self.lay_par[f"lay${i+1}_b"] = self.biases

    def for_prop(self):
        x_data = self.data
        for i in range(self.hidden_layers + 1):
            self.output = np.dot(x_data, self.lay_par[f"lay${i+1}_w"]) + self.lay_par[f"lay${i+1}_b"]
            x_data = self.layerActivation(i, self.output)
            self.lay_par[f"lay${i+1}_out"] = self.output
            self.lay_par[f"act_lay${i+1}_out"] = x_data


    # function to activate all layer outputs
    def layerActivation(self, i, input):
        #bassically we are making sure that if the iteration count gets to the number of hidden layers(we will be at the output later at that time)
        # so we want to use the right activation function for the output layer, else use the hiden layer activation function
        if i == self.hidden_layers:
            activated_out = act_softmax(input)
        else:
            activated_out = act_relu(input)
        return activated_out


    # function to calculate loss
    def cal_loss(self):
        return loss_fnc(pred_res=self.lay_par[f"act_lay${self.hidden_layers+1}_out"], act_res=self.label)


        # function to calculate gadient
    def back_prop(self):
        grads = {}
        m = self.data.shape[0]  # number of samples

        # --- Output layer gradient (softmax + cross-entropy) ---
        A_out = self.lay_par[f"act_lay${self.hidden_layers + 1}_out"]
        Y = self.label
        dZ = A_out.copy()
        dZ[range(m), Y] -= 1
        dZ /= m  # normalize

        # Loop backward through layers
        for i in reversed(range(1, self.hidden_layers + 2)):
            A_prev = self.data if i == 1 else self.lay_par[f"act_lay${i-1}_out"]
            W = self.lay_par[f"lay${i}_w"]

            # Compute gradients
            grads[f"dw{i}"] = np.dot(A_prev.T, dZ)
            grads[f"db{i}"] = np.sum(dZ, axis=0, keepdims=True)

            # Propagate to previous layer
            if i > 1:
                dA_prev = np.dot(dZ, W.T)
                Z_prev = self.lay_par[f"lay${i-1}_out"]
                dZ = dA_prev * self.relu_deriv(Z_prev)

        return grads


    # updating parameters
    def update_param(self, lr):
        gradients = self.back_prop()

        for i in range(self.hidden_layers +1):
            self.lay_par[f"lay${i+1}_w"] -= (lr * gradients[f"dw{i+1}"])
            self.lay_par[f"lay${i+1}_b"] -= (lr * gradients[f"db{i+1}"])

    # function to train
    def train_model(self, iterations, lr):
        self.layerInitialising(self.nue_nums)

        for i in range(iterations):
            self.for_prop()
            self.update_param(lr)
            loss = self.cal_loss()
            self.loss_by_epochs.append(loss)