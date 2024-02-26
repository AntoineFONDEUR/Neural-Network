#include <iostream>
#include "neural_network.h"
#include "loader.h"

int main(){
    NeuralNetwork<double> model(784,'b');
    model.add_layer(128, 'r');
    model.add_layer(64, 'r');
    model.add_layer(1, 's');

    Matrice<double> X_train;
    load("/home/antoine/Documents/info/CPP/neural_network/data/X_train.txt", X_train);
    Matrice<double> Y_train;
    load("/home/antoine/Documents/info/CPP/neural_network/data/Y_train.txt", Y_train);
    Matrice<double> X_test;
    load("/home/antoine/Documents/info/CPP/neural_network/data/X_test.txt", X_test);
    Matrice<double> Y_test;
    load("/home/antoine/Documents/info/CPP/neural_network/data/Y_test.txt", Y_test);

    /*std::ofstream outputFile("prediction.txt", std::ios::out);

    for (int i=0; i<1000; i++){
        NeuralNetwork<double> model(784,'b');
        model.add_layer(128, 'r');
        model.add_layer(64, 'r');
        model.add_layer(1, 's');

        //Matrice<double> test(X_test.p, 1, true, 0, 1);
        //Matrice<double> test = X_test.subMatrice(i,0,i+1,X_test.p);
        //test.reshape(X_test.p, 1);
    }

    outputFile.close();*/

    //model.test(X_test.subMatrice(0,0,100,X_test.p), Y_test.subMatrice(0,0,100,Y_test.p));

    //std::cout << model.get_cost(X_test.subMatrice(0,0,N,X_test.p), Y_test.subMatrice(0,0,N,Y_test.p)) << std::endl;

    int N=2000;
    Matrice<double> X_sample = X_test.subMatrice(0,0,N,X_test.p);
    Matrice<double> Y_sample = Y_test.subMatrice(0,0,N,Y_test.p);
    std::cout << "Positive prediction rate = " << model.test(X_sample, Y_sample) << std::endl;
    model.fit(X_train, Y_train, 1e-2, 3, 64, 3, 1,true);
    std::cout << "Positive prediction rate = " << model.test(X_sample, Y_sample) << std::endl;

    return 0;
}