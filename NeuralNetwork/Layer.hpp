#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>
#include <string>

#include "Neuron.hpp"

class Layer {

private:
    // Attribute Variables
    std::vector<Neuron> neurons;
    std::string activationFuncString;

    double (*activationFunc)(double) = nullptr;
    double (*d_activationFunc)(double) = nullptr;

    void assignActivationFunction(std::string& activationFuncString);

public:
    
    // Constructor
    Layer(
        int numNeurons,
        int numInputsPerNeuron,
        double learningRate,
        std::string activationFuncString
    );

    // Functional Methods
    std::vector<double> activateLayer(
        const std::vector<double> &inputs
    );

    void updateLayerWeightsAndBiases(
        const std::vector<double>& inputs, 
        const std::vector<double>& deltas
    );

    std::vector<Neuron>& getNeurons();

    // Utility Methods
    void printLayerWeightsAndBiases();

    void printNeuronOutputs();

    void printActivationFunc();

    double (*getActivationFunction())(double);

    double (*getActivationFunctionDerivative())(double);
};

#endif // LAYER_HPP