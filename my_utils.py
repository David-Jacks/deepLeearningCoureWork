#making neccesary imports

import pickle as pk
import numpy as np

# function to to extract data from dataset and filter samples to contain only the first 5 classes.
def filterSample(data_path: str):
    with open(data_path, "rb") as fo:
        img_dict = pk.load(fo, encoding="bytes")
    image_data = img_dict[b'data']
    image_label = img_dict[b'labels']
    new_img_data = []
    new_img_label = []
#   myDict = {}
# and then filter it so that i deal with features relating to the the label 0, 1, 2, 3, 4
    for i in range(len(image_label)):
        if image_label[i] <= 4:
            new_img_data.append(image_data[i])
            new_img_label.append(image_label[i])
    return (np.array(new_img_data), new_img_label)

 

# function to calculate the derivative of the reLU activation function
def relu_deriv(input):
    return np.array(input > 0, dtype=np.float32)

# this is the rectifield linear unit activation function
def act_relu(input):
    return np.maximum(input, 0)

# this function is used to calculate the accuracy of a model
def model_accuracy(output_layer, label):
    acc = np.argmax(output_layer, axis=1) == label
    return np.mean(acc) * 100

#softmax activation function implementatio
def act_softmax(input):
    # subtract the highest sample value in the output from all the other values in the sample.
    new_output = input - np.max(input, axis=1, keepdims=True)

    # exponentiating to make all outputs positive, since we are expecting a probability distrubution outcome
    exp_output = np.exp(new_output)

    # normalising the data
    normalised_output = exp_output / np.sum(exp_output, axis=1, keepdims=True)
    return normalised_output

#function to calculate the loss of a model
def loss_fnc(pred_res, act_res):
    # we want to make sure to prevent the loss getting to 0, or else it will affect the learning
    # so we clip it
    pred_res = np.clip(pred_res, 1e-7, 1-1e-7)

    loss_value = -np.log(pred_res[range(len(pred_res)), act_res])
    return np.mean(loss_value)