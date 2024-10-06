#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>

#include "readMNIST.hpp"
#include "../NeuralNetwork/NeuralNetwork.hpp"


// Functions for loading dataset:
void splitDataset(
    const std::vector<std::vector<double>>& dataset,
    std::vector<std::vector<double>>& trainContainer,
    std::vector<std::vector<double>>& testContainer,
    float ratio
){
    int nTrain = (int)(dataset.size() * ratio);

    int i = 0;
    while (i < nTrain) {
        trainContainer.push_back(dataset[i]);
        i++;
    }

    while (i < dataset.size()) {
        testContainer.push_back(dataset[i]);
        i++;
    }
}

int outputToLabel(std::vector<double>& nnOutput) {

    auto maxElement = std::max_element(nnOutput.begin(), nnOutput.end());
    int index = std::distance(nnOutput.begin(), maxElement);

    return index;
}

 // Testing functions:

double testNeuralNet(
    NeuralNetwork& nn,
    std::vector<std::vector<double>>& testInputData,
    std::vector<std::vector<double>>& testOutputData
) {

    std::vector<bool> outcomes;

    for (int i = 0; i < testInputData.size(); ++i) {
        
        std::vector<double> input = testInputData[i];
        std::vector<double> output = testOutputData[i];

        std::vector<double> prediction = nn.predict(input);
        
        int predictedLabel = outputToLabel(prediction);
        int actualLabel = outputToLabel(output);

        bool correct = (predictedLabel == actualLabel);
        outcomes.push_back(correct);
    }
    
    double accuracy = std::count(outcomes.begin(), outcomes.end(), true);
    accuracy /= outcomes.size();

    return accuracy;
}

std::pair<double, double> repeatTests(
    NeuralNetwork& nn,
    std::vector<std::vector<double>>& testInputData,
    std::vector<std::vector<double>>& testOutputData,
    int numRepeats
) {
    // testing prediction
    std::vector<double> accuracies;
    for (int i = 0; i < numRepeats; ++i) {
        double accuracy = testNeuralNet(
            nn,
            testInputData,
            testOutputData
        );
        accuracies.push_back(accuracy);
        std::cout << "accuracy: " << accuracy;
    }

    double meanAccuracy = std::accumulate(
        accuracies.begin(), 
        accuracies.end(), 
        0.0
    ) / accuracies.size();

    // Calculate standard deviation
    double sq_sum = std::inner_product(
        accuracies.begin(), 
        accuracies.end(),
        accuracies.begin(),
        0.0
    );

    double stddev = std::sqrt(sq_sum / accuracies.size() - meanAccuracy * meanAccuracy);

    return {meanAccuracy, stddev};
}


int main() {

    // read MNIST
    std::string pathMNIST = "Datasets/MNIST/mnist_train.csv";

    std::vector<std::vector<double>> mnistInputs, mnistOutputs;
    readMNIST(
        pathMNIST, mnistInputs, mnistOutputs, 60'000
    );
    std::cout << "Lines read: " << mnistInputs.size() << std::endl;

    // split into test and train.
    float splitFraction = 0.8;
    std::vector<std::vector<double>> trainInputs, testInputs, trainOutputs, testOutputs;
    splitDataset(mnistInputs, trainInputs, testInputs, splitFraction);
    splitDataset(mnistOutputs, trainOutputs, testOutputs, splitFraction);

    // create neural network
    std::vector<int> topology = {
        784, 128, 64, 10
    };

    std::vector<std::string> activationFuncTopology = {
        "sigmoid", "sigmoid", "sigmoid", "SoftMax"
    };

    NeuralNetwork nn(topology, activationFuncTopology, 0.15);
    
    std::cout << "\nnn created successfully." << "\nTraining:" << std::endl;

    // train neural netork on the given data, n epochs
    nn.train(
        trainInputs,
        trainOutputs,
        10
    );

    std::cout << "Training Complete." << std::endl;

    // test the resulting network using the repeatTests function  
    std::cout << "\nNow testing:" << std::endl;

    auto [meanAccuracy, stddev] = repeatTests(nn, testInputs, testOutputs, 5);

    std::cout << "Testing complete. Mean accuracy: " << 
            meanAccuracy << " std dev: " << stddev <<  std::endl; 

    return 0;
}
