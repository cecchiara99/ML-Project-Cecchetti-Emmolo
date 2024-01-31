# Project Title

Progect for Machine Learning masted degree course at University of Pisa

## Overview

- [Description](#description)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Authors](#authors)


## Description

This project is about the implementation from scratch of a Multi Layer Perceptron simulator in Python. The training phase consists in a Minibatch Stochastic Gradient Descent with Backpropagation. We perform a model selection exploiting Grid Search and a 5-Fold Cross Validation to test the network on the MONK problem and to find the best model for the CUP competition.
The final goal was obtain the output results on a Blind Test.


### Dependencies

This project dependencies are:
numpy: https://numpy.org/doc/1.26/
pandas: https://pandas.pydata.org/docs/
matplotlib: https://matplotlib.org/stable/index.html
itertools: https://docs.python.org/3/library/itertools.html
copy: https://docs.python.org/3/library/copy.html


### Installation

Install Python:
sudo apt install python3

Install pip:
sudo apt install --upgrade python3-pip

Install requirements:
python -m pip install --requirement requirements.txt


## Usage

python ./main.py

Note: in the file main.py is possible to modify the type of dataset and the relative task, the activaction functions to use for the hidden units and output units respectively, and the type of model selection.


## Authors

Cecchetti Chiara (c.cecchetti@studenti.unipi.it)
Emmolo Nicola (n.emmolo@studenti.unipi.it)

