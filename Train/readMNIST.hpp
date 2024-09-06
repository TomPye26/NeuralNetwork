#ifndef READMNIST_HPP
#define READMNIST_HPP

#include <vector>
#include <string>

// Function declaration
void readMNIST(
    const std::string& fileName,
    std::vector<std::vector<double>>& inputs,
    std::vector<std::vector<double>>& outputs
);

#endif // READMNIST_HPP
