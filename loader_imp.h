#include <fstream>
#include "loader.h"

void get_size_data(std::string path, int* n, int* p){
    std::ifstream inputFile(path);
    if (inputFile.is_open()) {
        std::string line;
        int i=0;
        while (std::getline(inputFile, line)) {
            if (i == 0){
                *n = std::stoi(line);
            }
            if (i == 1){
                *p = std::stoi(line);
                break;
            }
            i++;
        }
        inputFile.close();
    } else {
        std::cerr << "Unable to open the file." << std::endl;
    }
}

template <typename T> void load_array(std::string path, T* myarray, int n, int p){
    std::ifstream inputFile(path);
    
    if (inputFile.is_open()) {
        std::string line;
        int i=0;
        while (std::getline(inputFile, line)) {
            if (i>=2 && i-2<n*p){
                myarray[i-2] = std::stoi(line);
            }
            i++;
        }
        inputFile.close();
    } else {
        std::cerr << "Unable to open the file." << std::endl;
    }
}

template <typename T> void load(std::string path, Matrice<T>& data_mat){
    int n=0;
    int p=0;
    get_size_data(path, &n, &p);

    T* data = new T[n*p];
    load_array(path, data, n,p);
    data_mat = Matrice<T>(data, n, p);
    delete[] data;
}
