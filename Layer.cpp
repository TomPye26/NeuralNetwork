#include<vector>
#include<iostream>
#include<cmath>
#include<cstdlib>

#include"activationFunctions.hpp"
#include"Neuron.hpp"

class Layer {

public:

    // Attribute Variables
    std::vector<Neuron> neurons;

    // Constructor
    Layer(int numNeurons, int numInputsPerNeuron, double learningRate) {

        // setting random seed
        std::srand(std::time(nullptr));

        for (int i = 0; i < numNeurons; ++i) {
            Neuron n(numInputsPerNeuron, learningRate);
            neurons.push_back(n);
        }
    }

    // Functional Methods
    std::vector<double> activateLayer(const std::vector<double> &inputs, double (*activationFunc)(double)) {
        std::vector<double> outputs;
        for (int i = 0; i < neurons.size(); ++i) {
            Neuron& n = neurons[i];
            double n_activation_val = n.activate(inputs, activationFunc);

            outputs.push_back(n_activation_val);
        };

        return outputs;
    };

    void updateLayerWeightsAndBiases(const std::vector<double>& inputs, const std::vector<double>& deltas) {
        
        for (size_t i = 0; i < neurons.size(); ++i) {
            neurons[i].updateWeightsAndBias(inputs, deltas[i]);
        }
    }

    // Utility Methods

    void printLayerWeightsAndBiases() {
        for (int i = 0; i < neurons.size(); ++i) {
            std::cout << "Neuron " << i+1 << "\n"; 
            Neuron& n = neurons[i];
            n.printWeightsAndBias();
            std::cout << "\n";
        }
    }

};


int main() { 

    // example input
    std::vector<double> inputs = {0.5, -0.5, 1.0, -1.0};
    
    int numInputs = inputs.size();
    int numNeurons = 8; // arbitrary value
    double learningRate = 2.0;
    
    // create a layer
    Layer layer(numNeurons, numInputs, learningRate);

    // activate each neuron in the layer
    std::vector<double> outputs = layer.activateLayer(inputs, sigmoid);


    // update weights and biases (using exampe deltas)
    std::vector<double> deltas(8, 2.0);
    layer.updateLayerWeightsAndBiases(inputs, deltas);


    return 0;
}

