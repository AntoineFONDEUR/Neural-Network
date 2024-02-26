#ifndef LAYER_H
#define LAYER_H
#include "matrice.h"

template <typename T> class Layer;

template <typename T> T sigmoid(T z);
template <typename T> T sigmoid_deriv(T z);
template <typename T> T relu(T z);
template <typename T> T relu_deriv(T z);

template <typename T> class Layer{
    public :
        int nodes;
        Matrice<T> W;
        Matrice<T> gamma;
        Matrice<T> beta;
        Matrice<T> X;
        Matrice<T> Z_chap;
        Matrice<T> moy;
        Matrice<T> var;
        Matrice<T> Z;
        Matrice<T> Z_BN;

        Layer(int nodes, int input_size, char func_char);
        Matrice<T> compute(Matrice<T> X);
        Matrice<T> get_Z(Matrice<T> X);
        friend T sigmoid<T>(T z);
        friend T sigmoid_deriv<T>(T z);
        friend T relu<T>(T z);
        friend T relu_deriv<T>(T z);

        T (*activation_function)(T);
        T (*activation_func_deriv)(T);
};

#include "layer_imp.h"

#endif