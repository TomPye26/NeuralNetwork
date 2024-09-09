
#include<iostream>
#include<cmath>
#include<cstdlib>
#include<vector>

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
    if (x >= 0.0) {
        return x;
    } else {
        return 0.0;
    }
}

double d_ReLU(double x) {
    if (x >= 0.0) {
        return 1.0;
    } else {
        return 0.0;
    }
}

// leaky ReLU
double leakyReLU(double x){
    if (x >= 0.0) {
        return x;
    } else {
        return 0.1 * x;
    }
}

double d_leakyReLU(double x) {
    if (x >= 0.0) {
        return 1.0;
    } else {
        return 0.1;
    }
}


// softMax
std::vector<double> softMax(std::vector<double> inputVector) {


    double expSum = 0.0;
    for (const double x : inputVector) {
        expSum += std::exp(x);
    }

    for (int i = 0; i < inputVector.size(); ++i) {
        inputVector[i] = std::exp(inputVector[i]) / expSum; 
    }

    return inputVector;
}

double doNothing(double x) {
    return x;
}

double d_doNothing(double x) {
    return 1.0;
}