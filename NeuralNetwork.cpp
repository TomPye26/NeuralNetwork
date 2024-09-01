#include<vector>
#include<iostream>
#include<cmath>
#include<cstdlib>
#include<algorithm>

#include"activationFunctions.hpp"
#include"Layer.hpp"

class NeuralNetwork {
public:

    // Class Attributes
    std::vector<Layer> layers;
    std::vector<int> topology;

    // Constructor
    NeuralNetwork(const std::vector<int> &topology, double learningRate) : topology(topology) { 
        

        // starting at 1 as 0 is the input layer
        for (size_t i = 1; i < topology.size(); ++i) {
            // create layer for the given topology
            int numNueorns = topology[i];
            int inputsPerNeuron = topology[i-1];
            Layer layer(numNueorns, inputsPerNeuron, learningRate);
            
            layers.push_back(layer);
        }
    }

    // Functional Methods

    std::vector<std::vector<double>> propagateFowrards(
        const std::vector<double>& inputs,
        double (*activationFunc)(double)) {

        // OTT commenting to explain to myself...
        // vector of vectors giving the input to each neuron in each layers
        std::vector<std::vector<double>> layerValues;
        // input to each layer, initalised to the network's input
        std::vector<double> currentInputs = inputs;
        // initally, add the inputs of the first layer (network inputs) to this list
        layerValues.push_back(currentInputs);

        for (size_t i = 0; i < layers.size(); ++i) {
            
            currentInputs = layers[i].activateLayer(currentInputs, activationFunc);

            // apply softmax to the last layer
            if (i == layers.size()-1) {
                currentInputs = softMax(currentInputs);
            }

            layerValues.push_back(currentInputs);

            }
        
        return layerValues;
    }

    void propagateBackwards(

        // OTT commenting for my own benefit.

        const std::vector<std::vector<double>>& layerInputs,
        const std::vector<double>& expectedOutputs,
        double (*d_activationFunc)(double)) {


        // output layer is the last layer
        const std::vector<double>& outputLayer = layerInputs.back();

        // calculate loss and deltas for output layer
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
            Layer& nextLayer = layers[i + 1];

            // looping through nerons in layer i
            for (size_t j = 0; j < thisLayerOutput.size(); ++j) {
                
                double deltaSum = 0.0;
                for (size_t k = 0; k < nextLayer.neurons.size(); ++k) {

                    // weight connecting the neuron j from layer i 
                    // to neuron k in next layer
                    double weight_kj = nextLayer.neurons[k].weights[j];

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


    void train(
        const std::vector<std::vector<double>>& inputData,
        const std::vector<std::vector<double>>& targetOutputData,
        int numEpochs,
        double (*activationFunc)(double),
        double (*d_activationFunc)(double)
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
                std::vector<std::vector<double>> layerInputs = propagateFowrards(input, activationFunc);

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

                propagateBackwards(layerInputs, targetOutputs, d_activationFunc);

            }

            double averageLoss = totalLoss / inputData.size();

            std::cout << "Epoch " << epoch << " AverageLoss: " << averageLoss << std::endl;

            // Layer lastLayer = layers.back();
            // lastLayer.printNeuronOutputs();
        }
        
    }
};


int main() {

    // dummy data 
    std::vector<std::vector<double>> inputs = {{0.5, -0.5, 1.0, -1.0, 0.0}, {0.1, -0.3, 1.0, -1.0, 0.0}, {0.2, -0.4, 1.0, -1.0, 0.0}};
    std::vector<std::vector<double>> expectedOutputs = {{0.1, 0.9, 1.0}, {0.3, 0.9, 1.0}, {0.4, 0.9, 1.0}};
    
    std::vector<int> topology = {(int)inputs[0].size(), 4, 4, (int)expectedOutputs[0].size()};

    NeuralNetwork network(topology, 0.1);

    // std::cout << "\n\n" << std::endl;
    network.train(inputs, expectedOutputs, 100, sigmoid, d_sigmoid);

    return 0;
};