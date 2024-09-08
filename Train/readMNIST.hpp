#ifndef READMNIST_HPP
#define READMNIST_HPP

#include <vector>
#include <string>

// Function declaration
void readMNIST(
    const std::string& fileName,
    std::vector<std::vector<double>>& inputs,
    std::vector<std::vector<double>>& outputs,
    const int nRows
);
/**
 * Note: if nRows if larger than the number of rows in the file, 
 * 
 */

#endif // READMNIST_HPP
