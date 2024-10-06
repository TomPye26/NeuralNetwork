

## About The Project

A simple multi-layer-perceptron built in C++, using only the STL. 

I built this project to test my understanding of the basics of nerual networks, whilst practicing some C++.

## Project Roadmap
### Done so far:
- Implemented basic MLP.
- Create any topology.
- Chose any activation function for each layer.

### Ideas to come:
- Save and load trained models.
- Add different types of layers; namely convolutional layers, max pooling layers.
- Create and train own dataset of images.
- Visualistion.

## Example Usage
To test the network, let's train it on the MNIST dataset (original, I know). 

Within the *train* folder is the main training file *train.cpp*. Within it are some functions for loading and transforming the MNIST dataset from a csv. The main function loads and transforms the MNIST data, creates a neural network with the specified topology, runs the neural networks training loop over ``n`` epochs, and finally uses the ```predict``` method to test the overall acurracy on the testing dataset.

MNIST datasource: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download&select=mnist_train.csv

### Build commands:
Using g++ compiler:

```>> g++ -o TrainNetwork .\Train*.cpp .\NeuralNetwork*.cpp TrainNetwork.exe```

```>> .\TrainNetwork.exe```

### Output:
```Loading dataset
File Read.
Lines read: 60000

nn created successfully.
Training:
Epoch 0 Progress: 48000/48000 Done. AverageLoss: 0.0146224
Epoch 1 Progress: 48000/48000 Done. AverageLoss: 0.00814362
Epoch 2 Progress: 48000/48000 Done. AverageLoss: 0.00676755
Epoch 3 Progress: 48000/48000 Done. AverageLoss: 0.00571105
Epoch 4 Progress: 48000/48000 Done. AverageLoss: 0.00502699
Epoch 5 Progress: 48000/48000 Done. AverageLoss: 0.00468463
Epoch 6 Progress: 48000/48000 Done. AverageLoss: 0.00410649
Epoch 7 Progress: 48000/48000 Done. AverageLoss: 0.00401852
Epoch 8 Progress: 48000/48000 Done. AverageLoss: 0.00375962
Epoch 9 Progress: 48000/48000 Done. AverageLoss: 0.00374851
Training Complete.

Now testing.
Testing complete. Overall testing accuracy: 0.962167
```
96.2% accuracy... not too bad considering the simplicity of the network. Future work on adding convolutional and pooling layers would improve this further.