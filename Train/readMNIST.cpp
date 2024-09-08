#include "readMNIST.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

void readMNIST(
    const std::string& fileName,
    std::vector<std::vector<double>>& inputs,
    std::vector<std::vector<double>>& outputs,
    const int nRows
) {
    std::ifstream csvFile(fileName);
    std::string line;

    if (!csvFile.is_open()) {
        std::cerr << "Failed to read file " << fileName << std::endl;
        return;
    }

    // Skipping header
    csvFile && std::getline(csvFile, line);
        
    std::cout << "Loading dataset" << std::endl;


    int nLinesRead = 0;

    while (csvFile && std::getline(csvFile, line) && (nLinesRead < nRows)) {
        std::stringstream ss(line);
        std::string item;

        // Label data
        std::getline(ss, item, ',');
        int label = std::stoi(item);

        // Converting label to output vector
        // e.g. 3 -> {0, 0, 0, 1, 0, 0, 0  0, 0, 0}
        std::vector<double> output(10, 0.0);
        output[label] = 1.0;
        outputs.push_back(output);

        // Image data
        std::vector<double> input;
        while (std::getline(ss, item, ',')) {
            // Divide by 255 to normalize
            input.push_back(std::stod(item) / 255);
        }
        inputs.push_back(input);

        nLinesRead++;
    }

    csvFile.close();
    std::cout << "File Read." << std::endl;
}


// int main() {

//     std::vector<std::vector<double>> inputs, outputs;

//     std::string path = "C:\\Users\\thoma\\Documents\\NeuralNetwork\\Datasets\\MNIST\\mnist_train.csv";

//     readMNIST(path, inputs, outputs);

//     std::cout << "Read " << inputs.size() << " lines." << std::endl;

//     return 0;
// }