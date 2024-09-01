#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <iomanip>
#include <tuple>

void readMNIST (
    const std::string fileName,
    std::vector<std::vector<double>>& inputs,
    std::vector<std::vector<double>>& outputs
) {
    std::ifstream csvFile(fileName);
    std::string line;

    if (!csvFile.is_open()) {
        std::cerr << "Failed to read file" << fileName << std::endl;
    }

    // skipping header
    if (csvFile && std::getline(csvFile, line)) {
        ;
    }
    std::cout << "loading dataset" << std::endl;

    while (csvFile && std::getline(csvFile, line)) {

        std::stringstream ss(line);
        std::string item;

        // label data
        std::getline(ss, item, ',');
        int label = std::stoi(item);

        // converting label to output vector
        // e.g. 3 -> {0, 0, 0, 1, 0, ...}
        std::vector<double> output(10, 0.0);
        output[label] = 1.0;
        outputs.push_back(output);

        // image data
        std::vector<double> input;
        while (std::getline(ss, item, ',')) {
            // divide 255 to normalise
            input.push_back(std::stod(item)/255);
        }
        inputs.push_back(input);
    }

    csvFile.close();

    std::cout << "File Read." << std::endl;

}


int main() {

    std::vector<std::vector<double>> inputs, outputs;

    readMNIST("Datasets/MNIST/mnist_train.csv", inputs, outputs);

    return 0;
}