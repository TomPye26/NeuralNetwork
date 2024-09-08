#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <vector>
#include <string>

#include "Layer.hpp"

class NeuralNetwork {

private:
    // Class Attributes
    std::vector<Layer> layers;
    std::vector<int> topology;
    std::vector<std::string> activationFuncTopology; 

    // functional methods
    std::vector<std::vector<double>> propagateFowrards(
        const std::vector<double>& inputs
     );

    void propagateBackwards(
        const std::vector<std::vector<double>>& layerInputs,
        const std::vector<double>& expectedOutputs
    );


public:

    // constructor
    
    NeuralNetwork(
        const std::vector<int> &topology,
        const std::vector<std::string> &activationFuncTopology, 
        double learningRate
    );
    
    // functional accessible methods
    void train(
        const std::vector<std::vector<double>>& inputData,
        const std::vector<std::vector<double>>& targetOutputData,
        int numEpochs
    );

    std::vector<double> predict(const std::vector<double>& input);


};


#endif // NEURALNETWORK_HPP
