import matplotlib.pyplot as plt
from network import network_model
from my_utils import filterSample, data_extractor



# my deep model will have 6 layers
hidden_layers = 6

# nuerons per layers will be stored in an array 
nueron_num_array = [3072, 300, 250, 210, 180, 100, 50, 5 ]

#X_data, Y_data are the features and labels respectively
X_data, Y_data = filterSample(data_path="")

# instantiating my model
my_deep_model = network_model(X_train=X_data, Y_train=Y_data, no_hid_layer=hidden_layers, nueron_num_arr=nueron_num_array)