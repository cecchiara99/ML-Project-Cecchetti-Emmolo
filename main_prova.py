from read_data import *
from model_selection_assessment import *
from utils import *
import os

# Specify the paths to your training and test files for MONK's problems
path_file_train_1 = './monk+s+problems/monks-1.train'
path_file_train_2 = './monk+s+problems/monks-2.train'
path_file_train_3 = './monk+s+problems/monks-3.train'
path_file_test_1 = './monk+s+problems/monks-1.test'
path_file_test_2 = './monk+s+problems/monks-2.test'
path_file_test_3 = './monk+s+problems/monks-3.test'
# Specify the paths to your training and test files for CUP
path_file_train_cup = './cup+problem/ML-CUP23-TR.csv'
path_file_test_cup = './cup+problem/ML-CUP23-TS.csv'


# Specify the task you want to perform and the activation functions you want to use
task = "cup" # "monk1" or "monk2" or "monk3" or "cup"
activation_hidden = "relu" # depends on the task
activation_output = "identity" # depends on the task
type_selection = "k-fold" # "k-fold" or "hold-out"

data_X = None
data_y = None

if task == "monk1":
    data_X, data_y = read_monk(path_file_train_1)
    test_X, test_y = read_monk(path_file_test_1)
    print(f"\nLettura {task} completata\n")
elif task == "monk2":
    data_X, data_y = read_monk(path_file_train_2)
    test_X, test_y = read_monk(path_file_test_2)
    print(f"\nLettura {task} completata\n")
elif task == "monk3":
    data_X, data_y = read_monk(path_file_train_3)
    test_X, test_y = read_monk(path_file_test_3)
    print(f"\nLettura {task} completata\n")
elif task == "cup":
    data_X, data_y, test_X, test_y, blind_test_X = read_cup(path_file_train_cup, path_file_test_cup)
    print(f"\nLettura {task} completata\n")
else:
    print("Error: task not recognized")

input_size = data_X.shape[1]
output_size = data_y.shape[1]

print(f"\nTask: {task}\n")
print(f"Input size: {input_size}\n")  
print(f"Output size: {output_size}\n")
print(f"Activation hidden: {activation_hidden}\n")
print(f"Activation output: {activation_output}\n")
print(f"Type selection: {type_selection}\n")

# Train the model on the training set and select the best model -> last parameter is the type of model selection ('k-fold' o 'hold-out', default = 'k-fold')
best_results = model_selection(input_size, output_size, activation_hidden, activation_output, data_X, data_y, test_X, test_y, task, type_selection)

"""# Choose the best model (first of best result)
best_model = best_results[0]['model']

# Save the predictions on the blind test set for CUP
if task == "cup":
    predictions = best_model.predict(blind_test_X)
    create_cup_csv(predictions)"""


file_path = './learning_curve.png'
if os.path.exists(file_path):
    os.remove(file_path)

file_path = './accuracy_curve.png'
if os.path.exists(file_path):
    os.remove(file_path)

file_path = './mee_curve.png'
if os.path.exists(file_path):
    os.remove(file_path)
