#include "Layer.hpp"

#include <iostream>
#include <cmath>
#include <cstdlib>

#include"activationFunctions.hpp"


void Layer::assignActivationFunction(std::string& activationFuncString) {

    if (activationFuncString == "ReLU")
    {
        activationFunc = ReLU;
        d_activationFunc = d_ReLU;
    }  else if (activationFuncString == "sigmoid")
    {
        activationFunc = sigmoid;
        d_activationFunc = d_sigmoid;
    } else if (activationFuncString == "SoftMax")
    {
        /* when softmax, we are applying the function to the whole ouput
        vector, so don't need to do anything to individual neurons. */
        activationFunc = doNothing;
        d_activationFunc = d_doNothing;

    } else {
        std::cerr << "Unrecognised activation function: " << 
        activationFuncString << "\n Choose from: 'ReLU', 'sigmoid', " <<
        "'SoftMax'" << std::endl;

        return;
    }
    
}

// Constructor
Layer::Layer(
    int numNeurons,
    int numInputsPerNeuron,
    double learningRate,
    std::string activationFuncString
) : activationFuncString(activationFuncString) {

    // setting random seed
    std::srand(std::time(nullptr));

    for (int i = 0; i < numNeurons; ++i) {
        Neuron n(numInputsPerNeuron, learningRate);
        neurons.push_back(n);
    }

    assignActivationFunction(activationFuncString);
}

// Functional Methods
std::vector<double> Layer::activateLayer(
    const std::vector<double> &inputs
) {
    std::vector<double> outputs;
    
    for (int i = 0; i < neurons.size(); ++i) {
        Neuron& n = neurons[i];
        double n_activation_val = n.activate(inputs, activationFunc);

        outputs.push_back(n_activation_val);
    }

    if (activationFuncString == "SoftMax")
    {
        outputs = softMax(outputs);
    }
    

    return outputs;
};

void Layer::updateLayerWeightsAndBiases(
    const std::vector<double>& inputs, 
    const std::vector<double>& deltas
) {
    
    for (size_t i = 0; i < neurons.size(); ++i) {
        neurons[i].updateWeightsAndBias(inputs, deltas[i]);
    }
}

// Utility Methods
void Layer::printLayerWeightsAndBiases() {
    for (int i = 0; i < neurons.size(); ++i) {
        std::cout << "Neuron " << i+1 << "\n"; 
        Neuron& n = neurons[i];
        n.printWeightsAndBias();
        std::cout << "\n";
    }
}

void Layer::printNeuronOutputs() {
    for (int i = 0; i < neurons.size(); ++i) {
        std::cout << "Neuron " << i + 1 << " output: " << 
        neurons[i].output << std::endl;
    }
}

void Layer::printActivationFunc(){
    std::cout << "Activation Func: " << Layer::activationFuncString << std::endl;
}


int main() { 

    // example input
    std::vector<double> inputs = {0.5, -0.5, 1.0, -1.0};
    
    int numInputs = inputs.size();
    int numNeurons = 4; // arbitrary value
    double learningRate = 0.1;
    std::string activationFuncString = "SoftMax";
    
    // create a layer
    Layer layer(numNeurons, numInputs, learningRate, activationFuncString);

    // activate each neuron in the layer
    std::vector<double> outputs = layer.activateLayer(inputs);


    // update weights and biases (using exampe deltas)
    std::vector<double> deltas(numNeurons, 0.2);
    layer.updateLayerWeightsAndBiases(inputs, deltas);

    layer.printLayerWeightsAndBiases();

    layer.printActivationFunc();

    for (double o : outputs) {
        std::cout << o << "\n";
    }

    return 0;
}

