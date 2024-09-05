#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>

// #include"activationFunctions.hpp"

class Neuron {
public:

    // Attribute Variables
    std::vector<double> weights;
    double bias = 0.0;
    double output = 0.0;
    double learningRate = 0.1;

    // Contructor Method
    Neuron(int numInputs, double inputLearningRate)
    {

        // setting random seed
        // std::srand(std::time(nullptr));


        // Initalising other attributes
        learningRate = inputLearningRate;
        output = 0.0;

        // Initialsing weights and biases with random numbers
        for (int i_ = 0; i_ < numInputs; ++i_) {
            double rand_weight = (((double)std::rand() / RAND_MAX) * 2) -1;
            weights.push_back(rand_weight);
        }

        double rand_bias = (((double)std::rand() / RAND_MAX) * 2) -1;
        bias = rand_bias;

    };

    // Utility Methods
    void printWeightsAndBias(){
        for (int i = 0; i < weights.size(); ++i) {
            std::cout << "Weight " << i + 1 << ": " << weights[i] << "\n";
        }

        std::cout << "Bias: " << bias << std::endl;
    }

    // Required Functional Methods
    double activate(
        const std::vector<double> &inputs, 
        double (*activationFunc)(double)) {
        
            float sum = 0.0;
            // sum = W . I + B
            for (size_t i = 0; i < inputs.size(); ++i) {
                sum += inputs[i] * weights[i];
                // std::cout << "input["<<i<<"] = " << inputs[i] << std::endl;
                // std::cout << "weights["<<i<<"] = " << weights[i] << std::endl;
            };

            // std::cout << "bias " << bias << std::endl;
            sum += bias;

            // Squishification * with activation function
            // * Thank you 3Blue1Brown
            output = activationFunc(sum);

            return output;
    };

    void updateWeightsAndBias(const std::vector<double>& inputs, double delta) { 

        // uncomment the print parts for debugging.

        // std::cout << "Before update - Weights: ";
        // for (double weight : weights) std::cout << weight << " ";
        // std::cout << "\nBias: " << bias << "\n";
       
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] += learningRate * delta * inputs[i];
        };

        bias += learningRate * delta;

        // std::cout << "After update - Weights: ";
        // for (double weight : weights) std::cout << weight << " ";
        // std::cout << "\nBias: " << bias << "\n";

        // std::cout << "\n\n";
    }

private:

};

// int main() {

//     // Define input values
//     std::vector<double> inputs = {0.5, 1, -0.5, 0};

//     // Create neuron
//     Neuron n(inputs.size(), 0.1);

//     std::cout << "Initialising neuron with weights and biases:" << "\n";
//     n.printWeightsAndBias();

//     // Activate the neuron
//     n.activate(inputs, ReLU); 

//     // Update weights and biases (given a delta determiend in back-prop)
//     double delta = 0.1;
//     n.updateWeightsAndBias(inputs, delta);

//     // print the neurons output value
//     std::cout << "Neuron Output: " << n.output << std::endl;

//     return 0;

// }