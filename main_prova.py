from read_data import *
from model_selection import *

# Specify the paths to your training and test files
path_file_train_1 = './monk+s+problems/monks-1.train'
path_file_train_2 = './monk+s+problems/monks-2.train'
path_file_train_3 = './monk+s+problems/monks-3.train'

path_file_train_cup = './cup+problem/ML-CUP23-TR.csv'

task = "monk1" # "monk1" or "monk2" or "monk3" or "cup"
data_X = None
data_y = None

if task == "monk1":
    data_X, data_y = read_monk(path_file_train_1)
elif task == "monk2":
    data_X, data_y = read_monk(path_file_train_2)
elif task == "monk3":
    data_X, data_y = read_monk(path_file_train_3)
elif task == "cup":
    data_X, data_y = read_cup(path_file_train_cup)
else:
    print("Error: task not recognized")

print("Input-shape: ", data_X.shape)
print("Targets-shape: ", data_y.shape)


input_size = data_X.shape[1]
output_size = data_y.shape[1]
activation_hidden = "sigmoid"
activation_output = "tanh"


# Train the model on the training set and select the best model
best_model = model_selection(input_size, output_size, activation_hidden, activation_output, data_X, data_y, K=5)

# DA SPOSTARE
"""
def model_assessment(final_model, test_data):
    # Return the test error
    pass
"""
