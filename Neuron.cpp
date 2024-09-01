#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>>

class Neuron {
public:

    // Attribute Variables
    std::vector<double> weights;
    double bias;
    double output;
    double learningRate;

    // Contructor Method
    Neuron(int numInputs, double learningRate) {

        // setting random seed
        std::srand(std::time(nullptr));

        // Initialsing weights and biases with random numbers
        for (int i = 0; i < numInputs; ++i) {
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
        
            double sum = 0.0;
            // sum = W . I + B
            for (size_t i = 0; i < inputs.size(); ++i) {
                sum += inputs[i] * weights[i];
            };

            sum += bias;

            // Squishification * with activation function
            // * Thank you 3Blue1Brown
            output = activationFunc(sum);

            return output;
    };

    void updateWeightsAndBias(const std::vector<double>& inputs, double delta) { 
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] += inputs[i] * learningRate * delta;
        };

        bias += learningRate + delta;
    }

private:

};

// different activation functions:

// sigmoid (and its derivative)
double sigmoid(double x) {
    double sig = 1.0 / (1.0 + std::exp(-x));
    return sig;
}

double d_sigmoid(double x) {
    double d_sig = x * (1.0 - x);
    return d_sig;
}

// ReLU
double ReLU(double x) {
    if (x >= 0) {
        return x;
    } else {
        return 0;
    }
}

double d_ReLU(double x) {
    if (x >= 0) {
        return 1;
    } else {
        return 0;
    }
}

int main() {

    // Define input values
    std::vector<double> inputs = {0.5, 1, -0.5, 0};

    // Create neuron
    Neuron n(inputs.size(), 0.1);

    std::cout << "Initialising neuron with weights and biases:" << "\n";
    n.printWeightsAndBias();

    // Activate the neuron
    n.activate(inputs, ReLU); 

    // Update weights and biases (given a delta determiend in back-prop)
    double delta = 0.1;
    n.updateWeightsAndBias(inputs, delta);

    // print the neurons output value
    std::cout << "Neuron Output: " << n.output << std::endl;

    return 0;

}