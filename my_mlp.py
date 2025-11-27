import matplotlib.pyplot as plt
from network import network_model
from my_utils import filterSample, model_accuracy

# this is the nueron arrays stating the number of neurons i will want to give to each layer
nueron_num_array = [250, 150, 85, 5]

# path to the cifar-10 dataset
cifer_data_path = "./cifar-10-batches-py/data_batch_1"
test_cifer_data_path = "./cifar-10-batches-py/test_batch"

#X_train_data, Y_train_data are the features and labels respectively for training
X_train_data, Y_train_data = filterSample(data_path=cifer_data_path)

#X_test_data, Y_test_data are the features and labels respectively for testing
X_test_data, Y_test_data = filterSample(data_path=test_cifer_data_path)


# instantiating my model (disable batch norm for MLP)
my_multilayer_model = network_model(X_train=X_train_data, Y_train=Y_train_data, no_hid_layer=3, nueron_num_arr=nueron_num_array, use_batchnorm=False)

#training
my_multilayer_model.train_model(iterations=100, lr=0.02, val_data=X_train_data[20:], val_label=Y_train_data[20:], batch_size=128)

# calculating accuracy
acc = my_multilayer_model.cal_accuracy()

# calculating loss
loss  = my_multilayer_model.cal_loss()

# printing shapes and results
print("shape of X_train_data is:", X_train_data.shape)

print(f"Model accuracy is: {acc}%")
print(f"Model loss is: {loss}")

# testing my model
X_test, Y_test = filterSample(data_path=test_cifer_data_path)
test_output = my_multilayer_model.test_model(X_test)

test_acc = model_accuracy(output_layer=test_output, label=Y_test)
print(f"Test accuracy is: {test_acc}%")
