import matplotlib.pyplot as plt
from network import network_model
from my_utils import filterSample, model_accuracy, plot_fn
import numpy as np


# my deep model will have 12 layers
hidden_layers = 12

# nuerons per layers will be stored in an array 
nueron_num_array = [300, 250, 205, 180, 100, 50, 70, 80, 50, 60, 90, 50, 5 ]

# path to the cifar-10 dataset and the test sample batch
cifer_data_path = "./cifar-10-batches-py/data_batch_1"
test_cifer_data_path = "./cifar-10-batches-py/test_batch"

#X_train_data, Y_train_data are the features and labels respectively for training
X_train_data, Y_train_data = filterSample(data_path=cifer_data_path)

#X_test_data, Y_test_data are the features and labels respectively for testing
X_test_data, Y_test_data = filterSample(data_path=test_cifer_data_path)

#setting the validation size to be 30% of the training data
val_size = int(0.3 * X_train_data.shape[0])
X_val = X_train_data[-val_size:]
Y_val = Y_train_data[-val_size:]
X_tr = X_train_data[:-val_size]
Y_tr = Y_train_data[:-val_size]

# instantiating my model using adam optimizer with milder regularization and dropout
my_deep_model = network_model(X_train=X_tr, Y_train=Y_tr, no_hid_layer=hidden_layers, nueron_num_arr=nueron_num_array, optimizer='adam', l2_lambda=0.001, dropout_rate=0.20)

#training with larger batch size
my_deep_model.train_model(iterations=100, lr=0.001, val_data=X_val, val_label=Y_val, batch_size=64, patience=5)

# calculating accuracy
acc = my_deep_model.cal_accuracy()

# calculating loss
loss  = my_deep_model.cal_loss()


print(f"Model accuracy is: {acc}%")
print(f"Model loss is: {loss}")

# testing my model
X_test, Y_test = filterSample(data_path=test_cifer_data_path)
test_output = my_deep_model.test_model(X_test)

test_acc = model_accuracy(output_layer=test_output, label=Y_test)
print(f"Test accuracy is: {test_acc}%")

# computing the confusion matrix
my_matrix = my_deep_model.conf_matr(test_output, Y_test)

print("Confusion Matrix:")
print(my_matrix)

# printing the training loss curve
train_loss_df = my_deep_model.gen_train_loss_curve()
val_loss_df = my_deep_model.gen_val_loss_curve()
# print(train_loss_df)
plot_fn(tr_df=train_loss_df, val_df=val_loss_df, model_type="Deep_Network")