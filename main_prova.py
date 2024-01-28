from read_data import *
from model_selection_assessment import *
from utils import *
import os

# Specify the paths to your training and test files
path_file_train_1 = './monk+s+problems/monks-1.train'
path_file_train_2 = './monk+s+problems/monks-2.train'
path_file_train_3 = './monk+s+problems/monks-3.train'
path_file_test_1 = './monk+s+problems/monks-1.test'
path_file_test_2 = './monk+s+problems/monks-2.test'
path_file_test_3 = './monk+s+problems/monks-3.test'

path_file_train_cup = './cup+problem/ML-CUP23-TR.csv'
path_file_test_cup = './cup+problem/ML-CUP23-TS.csv'

task = "monk1" # "monk1" or "monk2" or "monk3" or "cup"

print(f"\nTask: {task}\n")

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

print(f"Input size: {input_size}\n")  
print(f"Output size: {output_size}\n")

activation_hidden = "sigmoid"
activation_output = "tanh"
K = 5

print(f"Activation hidden: {activation_hidden}\n")
print(f"Activation output: {activation_output}\n")

type_selection = 'k-fold' # "k-fold" or "hold-out"

print(f"Type selection: {type_selection}\n")

# Train the model on the training set and select the best model -> ultimo parametro scegliere tipo model selection ('k-fold' o 'hold-out') DI DEFAULT è K-FOLD

best_model = model_selection(input_size, output_size, activation_hidden, activation_output, data_X, data_y, task, test_X, test_y, type_selection)

model_assessment(best_model, test_X, test_y, task)

task = 'monk2'
data_X, data_y = read_monk(path_file_train_2)
test_X, test_y = read_monk(path_file_test_2)
print(f"\nLettura {task} completata\n")

input_size = data_X.shape[1]
output_size = data_y.shape[1]

best_model = model_selection(input_size, output_size, activation_hidden, activation_output, data_X, data_y, task, test_X, test_y, type_selection)
model_assessment(best_model, test_X, test_y, task)

task = 'monk3'
data_X, data_y = read_monk(path_file_train_3)
test_X, test_y = read_monk(path_file_test_3)
print(f"\nLettura {task} completata\n")



best_model = model_selection(input_size, output_size, activation_hidden, activation_output, data_X, data_y, task, test_X, test_y, type_selection)
model_assessment(best_model, test_X, test_y, task)


if task == "cup":
    predictions = best_model.predict(blind_test_X)
    create_cup_csv(predictions)

file_path = './learning_curve.png'

if os.path.exists(file_path):
    os.remove(file_path)

file_path = './accuracy_curve.png'

if os.path.exists(file_path):
    os.remove(file_path)

file_path = './mee_curve.png'
if os.path.exists(file_path):
    os.remove(file_path)
