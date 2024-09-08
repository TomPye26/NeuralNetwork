#include "NeuralNetwork.hpp"

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <algorithm>



// Constructor
NeuralNetwork::NeuralNetwork(
    const std::vector<int> &topology,
    const std::vector<std::string> &activationFuncTopology, 
    double learningRate
) : topology(topology), activationFuncTopology(activationFuncTopology) { 
    

    // starting at 1 as 0 is the input layer
    for (size_t i = 1; i < topology.size(); ++i) {
        // create layer for the given topology
        int numNueorns = topology[i];
        int inputsPerNeuron = topology[i-1];
        std::string activationFuncString = activationFuncTopology[i];

        Layer layer(
            numNueorns, 
            inputsPerNeuron, 
            learningRate, 
            activationFuncString
        );
        
        layers.push_back(layer);
    }
}

// Functional Methods

std::vector<std::vector<double>> NeuralNetwork::propagateFowrards(
    const std::vector<double>& inputs
    ) {

    // OTT commenting to explain to myself...
    // vector of vectors giving the input to each neuron in each layers
    std::vector<std::vector<double>> layerValues;
    // input to each layer, initalised to the network's input
    std::vector<double> currentInputs = inputs;
    // initally, add the inputs of the first layer (network inputs) to this list
    layerValues.push_back(currentInputs);

    for (size_t i = 0; i < layers.size(); ++i) {
        
        currentInputs = layers[i].activateLayer(currentInputs);

        layerValues.push_back(currentInputs);

        }
    
    return layerValues;
}

void NeuralNetwork::propagateBackwards(
    const std::vector<std::vector<double>>& layerInputs,
    const std::vector<double>& expectedOutputs
) {

    // OTT commenting for my own benefit.   

    // output layer is the last layer
    const std::vector<double>& outputLayer = layerInputs.back();

    // calculate loss and deltas for output layer
    Layer& lastLayer = layers.back();
    double (*d_activationFunc)(double) = lastLayer.getActivationFunctionDerivative();

    std::vector<double> outputDeltas;
    for (size_t i = 0; i < outputLayer.size(); ++i) {
        double output_i = outputLayer[i];
        double loss = expectedOutputs[i] - output_i;
        
        // delta = loss * d(activationFunc)/dx
        
        double delta = loss * d_activationFunc(output_i);

        outputDeltas.push_back(delta);
    }

    // now calculate loss for each hidden layer

    // first add in deltas of output layer calculated above
    std::vector<std::vector<double>> layerDeltas;
    layerDeltas.push_back(outputDeltas);
    
    // looping backwards through hidden layers
    for (int i = layers.size() - 2; i >=0; --i) {
        std::vector<double> hiddenDeltas;
        std::vector<double> thisLayerOutput = layerInputs[i + 1];
        std::vector<double> nextLayerDeltas = layerDeltas.back();
        Layer& thisLayer = layers[i];
        double (*d_activationFunc)(double) = thisLayer.getActivationFunctionDerivative();
        Layer& nextLayer = layers[i + 1];
        std::vector<Neuron>& nextLayerNeurons = nextLayer.getNeurons(); 

        // looping through nerons in layer i
        for (size_t j = 0; j < thisLayerOutput.size(); ++j) {
            
            double deltaSum = 0.0;
            for (size_t k = 0; k < nextLayerNeurons.size(); ++k) {

                // weight connecting the neuron j from layer i 
                // to neuron k in next layer
                double weight_kj = nextLayerNeurons[k].weights[j];

                // delta of neuron k in the next layer.
                double delta_k = nextLayerDeltas[k];

                // Their product represents how much the error in the k-th neuron of the next layer 
                // influences the j-th neuron in the current layer.

                deltaSum += weight_kj * delta_k;
            }

            hiddenDeltas.push_back(deltaSum * d_activationFunc(thisLayerOutput[j]));
        }
        layerDeltas.push_back(hiddenDeltas);
    }
    
    // reverse layerDeltas as was computed backwards
    std::reverse(layerDeltas.begin(), layerDeltas.end());

    // update weights and biases 
    for (size_t i = 0; i < layers.size(); ++i) {
        layers[i].updateLayerWeightsAndBiases(layerInputs[i], layerDeltas[i]);
    }
}


void NeuralNetwork::train(
    const std::vector<std::vector<double>>& inputData,
    const std::vector<std::vector<double>>& targetOutputData,
    int numEpochs
) {
    for (int epoch = 0; epoch < numEpochs; ++epoch) {

        double totalLoss = 0.0;

        if (inputData[0].size() != topology[0]) {
            throw ("Num first layer neurons not equal to input size.");
        }

        for (size_t i = 0; i < inputData.size(); ++i) {
            
            // std::cout << "input data point " << i+1 << std::endl;

            std::vector<double> input = inputData[i];
            std::vector<double> targetOutputs = targetOutputData[i];

            // prop forward
            // std::cout << "forward prop" << std::endl;
            std::vector<std::vector<double>> layerInputs ;
            layerInputs= propagateFowrards(input);

            // calculate loss
            const std::vector<double>& outputLayer = layerInputs.back();
            double loss = 0.0;
            for (size_t j = 0; j < outputLayer.size(); ++j) {
                double error = targetOutputs[j] - outputLayer[j];
                loss += std::pow(error, 2);
            }
            
            totalLoss += loss / outputLayer.size();


            // prop backwards
            // std::cout << "backwards prop" << std::endl;

            propagateBackwards(layerInputs, targetOutputs);

        }

        double averageLoss = totalLoss / inputData.size();

        std::cout << "Epoch " << epoch << " AverageLoss: " << averageLoss << std::endl;

        // Layer lastLayer = layers.back();
        // lastLayer.printNeuronOutputs();
    }
    
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {

    std::vector<double> thisInput = input;

    // forwards pass returning only values of last layer
    for (size_t i = 0; i < layers.size(); ++i) {
        thisInput = layers[i].activateLayer(thisInput);
    }

    return thisInput;

};


// int main() {

//     // // dummy data 
//     std::vector<std::vector<double>> inputs = {{0.5, -0.5, 1.0, -1.0, 0.0}, {0.1, -0.3, 1.0, -1.0, 0.0}, {0.2, -0.4, 1.0, -1.0, 0.0}};
//     std::vector<std::vector<double>> expectedOutputs = {{0.0, 1.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}};
    
//     std::vector<int> topology = {
//         (int)inputs[0].size(), 8, 8, (int)expectedOutputs[0].size()
//     };

//     std::vector<std::string> actFuncTopology = {
//         "ReLU", "ReLU", "ReLU", "ReLU"
//     };

//     NeuralNetwork network(topology, actFuncTopology, 0.1);

//     // std::cout << "\n\n" << std::endl;
//     network.train(inputs, expectedOutputs, 200);

//     std::vector<double> prediction = network.predict(inputs[0]);
//     std::cout << "Prediction:" << std::endl;
//     for (double x : prediction) {
//         std::cout << x << std::endl;
//     }
//     return 0;

// };