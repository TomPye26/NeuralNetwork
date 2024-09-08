#include <vector>
#include <iostream>
#include <string>
#include <algorithm>

#include "readMNIST.hpp"
#include "../NeuralNetwork/NeuralNetwork.hpp"

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

    // std::cout << trainDataset.size() << std::endl;
    // std::cout << testDataset.size() << std::endl;
}

int outputToLabel(std::vector<double>& nnOutput) {

    auto maxElement = std::max_element(nnOutput.begin(), nnOutput.end());
    int index = std::distance(nnOutput.begin(), maxElement);

    return index;
}

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

        std::cout << "Predicted: " << predictedLabel << " Actual: " << actualLabel << "\n";

        bool correct = (predictedLabel == actualLabel);
        outcomes.push_back(correct);
        
    }
    
    double accuracy = std::count(outcomes.begin(), outcomes.end(), true);

    accuracy /= outcomes.size();

    return accuracy;

}


int main() {

    // read MNIST
    std::string pathMNIST = "Datasets/MNIST/mnist_train.csv";

    std::vector<std::vector<double>> mnistInputs, mnistOutputs;
    readMNIST(
        pathMNIST, mnistInputs, mnistOutputs, 20'000
    );
    std::cout << "Lines read: " << mnistInputs.size() << std::endl;

    // split into test and train.
    std::vector<std::vector<double>> trainInputs;
    std::vector<std::vector<double>> testInputs;
    splitDataset(mnistInputs, trainInputs, testInputs, 0.8);

    std::vector<std::vector<double>> trainOutputs;
    std::vector<std::vector<double>> testOutputs;
    splitDataset(mnistOutputs, trainOutputs, testOutputs, 0.8);

    // create neural network
    std::vector<int> topology = {
        784, 128, 128, 10
    };

    std::vector<std::string> activationFuncTopology = {
        "sigmoid", "sigmoid", "sigmoid", "SoftMax"
    };

    NeuralNetwork nn(topology, activationFuncTopology, 0.1);
    
    std::cout << "\nnn created successfully." << std::endl;

    nn.train(
        trainInputs,
        trainOutputs,
        10
    );

    // testing prediction
    double accuracy = testNeuralNet(
        nn,
        testInputs,
        testOutputs
    );

    std::cout << "Accuracy: " << accuracy << std::endl; 

    // std::vector<double> prediction = nn.predict(testInputs[25]);

    // std::cout << "\n\nactual: ";
    // for (double v: testOutputs[25]) {
    //     std::cout << v << " ";
    // }

    // std::cout << "\nprediction: ";
    // for (double v: prediction) {
    //     std::cout << v << " ";
    // }

    // std::cout << "\n" << outputToLabel(prediction) << std::endl; 

    return 0;
}
