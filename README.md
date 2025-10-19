# iris-week4-hw

This repo serves as the basic introduction to continuous integartion and delivery of machine learning models

The repo consists of the following files
dev branch has all of these files
requirements.txt - all python packages required for running the train and data version control
models - this folder contains the .joblib files which are result of the trained model and the encodings done to the data
train.py - contains the code used for training the data- a simple logistic regression trained on train test split data
tests - the folder contains the test (3 in this case) that were run as a part of the process
.github/workflows/sanity.yml - .github/workflows is the default location to create action files. sanity.yml contains the actions and steps taken in the action and cases that triggers the actions

