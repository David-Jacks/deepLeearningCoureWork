#making neccesary imports
import pickle as pk
import numpy as np
import os
import matplotlib.pyplot as plt

# path to the cifar-10 dataset and the test sample batch
# please make sure these paths are correct before running the code
# and do not worry about weather I am passing all the data since you can only see the data_batch_1 path here
# but inside the filterSample function I am making sure to collect all five training batches for training

cifer_data_path = "./cifar-10-batches-py/data_batch_1"
test_cifer_data_path = "./cifar-10-batches-py/test_batch"

# function to to extract data from dataset and filter samples to contain only the first 5 classes.
def filterSample(test: bool=False):
    if test:
        data_path = test_cifer_data_path
    else:
        data_path = cifer_data_path

    new_img_data = []
    new_img_label = []

    base_name = os.path.basename(data_path)

    # this path is for the single test batch in the dataset
    if "test_batch" in base_name:
        with open(data_path, "rb") as fo:
            img_dict = pk.load(fo, encoding="bytes")
        image_data = img_dict[b'data']
        image_label = img_dict[b'labels']
        for i in range(len(image_label)):
            if image_label[i] <= 4:
                new_img_data.append(image_data[i])
                new_img_label.append(image_label[i])

    # I want to make surei collect all five training data_batch_1..5 to use for training
    elif "data_batch" in base_name:
        dirn = os.path.dirname(data_path)
        # print(dirn)
        for idx in range(1, 6):
            file_path = os.path.join(dirn, f"data_batch_{idx}")
            with open(file_path, "rb") as fo:
                img_dict = pk.load(fo, encoding="bytes")
            image_data = img_dict[b'data']
            image_label = img_dict[b'labels']
            for j in range(len(image_label)):
                if image_label[j] <= 4:
                    new_img_data.append(image_data[j])
                    new_img_label.append(image_label[j])
    else:
        #trying to open the given path as a single batch file
        with open(data_path, "rb") as fo:
            img_dict = pk.load(fo, encoding="bytes")
        image_data = img_dict[b'data']
        image_label = img_dict[b'labels']
        for i in range(len(image_label)):
            if image_label[i] <= 4:
                new_img_data.append(image_data[i])
                new_img_label.append(image_label[i])

    # normalising my training data within the range of 0 to 1, because the pixel values ranges from 0 to 255
    new_img_data = np.array(new_img_data, dtype=np.float32) / 255
    new_img_label = np.array(new_img_label, dtype=np.int64)
    return new_img_data, new_img_label

 

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
    # I want to make sure to prevent the loss getting to 0, or else it will affect the learning
    # so I will clip the predicted responses from the output layer
    pred_res = np.clip(pred_res, 1e-7, 1-1e-7)

    loss_value = -np.log(pred_res[range(len(pred_res)), act_res])
    return np.mean(loss_value)


def plot_fn(tr_df=None, val_df=None, model_type:str=""):
    df_train = tr_df
    df_val = val_df

    if df_train is not None and not df_train.empty:
        plt.plot(df_train['epoch'], df_train['loss'], marker='o', linestyle='-', linewidth=1, label='train loss')
    if df_val is not None and not df_val.empty:
        plt.plot(df_val['epoch'], df_val['val_loss'], marker='x', linestyle='--', linewidth=1, label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curves for {model_type} Model')
    plt.legend()
    plt.grid(True)
    plt.show()


