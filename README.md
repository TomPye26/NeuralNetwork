MNIST datasource: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download&select=mnist_train.csv

#Build commands:
g++ -o TrainNetwork .\Train\*.cpp .\NeuralNetwork\*.cpp
TrainNetwork.exe
