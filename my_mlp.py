from network import network_model
from my_utils import filterSample, model_accuracy, plot_fn

# this is the nueron arrays stating the number of neurons i will want to give to each layer, it doesn't include the input layer nueron, 
# as I am calculating that implicitly and also every neuron size in a layer becomes an input size to the layer following it
nueron_num_array = [250, 150, 85, 5]

# path to the cifar-10 dataset
cifer_data_path = "./cifar-10-batches-py/data_batch_1"
test_cifer_data_path = "./cifar-10-batches-py/test_batch"

#X_train_data, Y_train_data are the features and labels respectively for training
X_train_data, Y_train_data = filterSample(data_path=cifer_data_path)

#X_test_data, Y_test_data are the features and labels respectively for testing
X_test_data, Y_test_data = filterSample(data_path=test_cifer_data_path)

# use last 5k samples as validation (if dataset is 25k samples)
val_size =  int(0.3 * X_train_data.shape[0])
print("Total training samples are:", X_train_data.shape[0])
print("Validation size is:", val_size)
X_val = X_train_data[-val_size:]
Y_val = Y_train_data[-val_size:]
X_tr = X_train_data[:-val_size]
Y_tr = Y_train_data[:-val_size]

# instantiating my model (disable batch norm for MLP)
my_multilayer_model = network_model(X_train=X_tr, Y_train=Y_tr, no_hid_layer=3, nueron_num_arr=nueron_num_array, use_batchnorm=False, dropout_rate=0)

#training
my_multilayer_model.train_model(iterations=100, lr=0.01, val_data=X_val, val_label=Y_val, batch_size=64, patience=5)

# calculating accuracy
acc = my_multilayer_model.cal_accuracy()

# calculating loss
loss  = my_multilayer_model.cal_loss()


# print("shape of X_train_data is:", X_train_data.shape)

print(f"Model accuracy is: {acc}%")
print(f"Model loss is: {loss}")

# testing my model
X_test, Y_test = filterSample(data_path=test_cifer_data_path)

# this the predictions we got from the model after presenting the test data to it
test_output = my_multilayer_model.test_model(X_test)

test_acc = model_accuracy(output_layer=test_output, label=Y_test)
print(f"Test accuracy is: {test_acc}%")


# computing the confusion matrix
my_matrix = my_multilayer_model.conf_matr(test_output, Y_test)

print("Confusion Matrix:")
print(my_matrix)

# printing the training loss curve
train_loss_df = my_multilayer_model.gen_train_loss_curve()
val_loss_df = my_multilayer_model.gen_val_loss_curve()
# print(train_loss_df)
plot_fn(tr_df=train_loss_df, val_df=val_loss_df, model_type="MLP")