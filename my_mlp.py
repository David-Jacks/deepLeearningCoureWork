import matplotlib.pyplot as plt
from network import network_model
from my_utils import filterSample

# this is the nueron arrays stating the number of neurons i will want to give to each layer
nueron_num_array = [3072, 300, 150, 80, 5]


#X_data, Y_data are the features and labels respectively
X_data, Y_data = filterSample(data_path="")

# instantiating my model
my_multilayer_model = network_model(X_train=X_data, Y_train=Y_data, no_hid_layer=3, nueron_num_arr=nueron_num_array)

#training
my_multilayer_model.train_model(iterations=1000, lr=0.02)

# calculating loss