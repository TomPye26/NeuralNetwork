#ifndef NEURON_HPP
#define NEURON_HPP

#include <vector>

// #include"activationFunctions.hpp"

class Neuron {

public:

    // Attribute Variables
    // maybe these should be private with access methods?
    std::vector<double> weights;
    double bias = 0.0;
    double output = 0.0;
    double learningRate = 0.1;


    // Contructor Method
    Neuron(int numInputs, double inputLearningRate);

    // Utility Methods
    void printWeightsAndBias();

    // Required Functional Methods
    double activate(
        const std::vector<double> &inputs, 
        double (*activationFunc)(double)
    );

    void updateWeightsAndBias(
        const std::vector<double>& inputs,
        double delta
    );


};


#endif // NEURON_HPP
