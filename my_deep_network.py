import matplotlib.pyplot as plt
from network import network_model
from my_utils import filterSample, model_accuracy



# my deep model will have 12 layers
hidden_layers = 12

# nuerons per layers will be stored in an array 
nueron_num_array = [300, 250, 205, 180, 100, 50, 70, 80, 50, 60, 90, 50, 5 ]

# path to the cifar-10 dataset
cifer_data_path = "./cifar-10-batches-py/data_batch_1"
test_cifer_data_path = "./cifar-10-batches-py/test_batch"

#X_train_data, Y_train_data are the features and labels respectively for training
X_train_data, Y_train_data = filterSample(data_path=cifer_data_path)

#X_test_data, Y_test_data are the features and labels respectively for testing
X_test_data, Y_test_data = filterSample(data_path=test_cifer_data_path)

# instantiating my model using adam optimizer
# Simple diagnostics and a proper train/validation split
import numpy as np
print("Full training set shape:", X_train_data.shape)
unique, counts = np.unique(Y_train_data, return_counts=True)
print("Label distribution (label:count):", dict(zip(unique.tolist(), counts.tolist())))

# use last 5k samples as validation (if dataset is 25k samples)
val_size = 5000 if X_train_data.shape[0] > 5000 else int(0.2 * X_train_data.shape[0])
X_val = X_train_data[-val_size:]
Y_val = Y_train_data[-val_size:]
X_tr = X_train_data[:-val_size]
Y_tr = Y_train_data[:-val_size]

# instantiating my model using adam optimizer with milder regularization and dropout
my_deep_model = network_model(X_train=X_tr, Y_train=Y_tr, no_hid_layer=hidden_layers, nueron_num_arr=nueron_num_array, optimizer='adam', l2_lambda=0.001, dropout_rate=0.20)

#training with larger batch size
my_deep_model.train_model(iterations=100, lr=0.001, val_data=X_val, val_label=Y_val, batch_size=256)

# calculating accuracy
acc = my_deep_model.cal_accuracy()

# calculating loss
loss  = my_deep_model.cal_loss()

# printing shapes and results
print("shape of X_train_data is:", X_train_data.shape)

print(f"Model accuracy is: {acc}%")
print(f"Model loss is: {loss}")

# testing my model
X_test, Y_test = filterSample(data_path=test_cifer_data_path)
test_output = my_deep_model.test_model(X_test)

test_acc = model_accuracy(output_layer=test_output, label=Y_test)
print(f"Test accuracy is: {test_acc}%")