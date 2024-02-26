#ifndef LOADER_H
#define LOADER_H
#include "matrice.h"

void get_size_data(std::string path, int* n, int* p);
template <typename T> void load_array(std::string path, T* myarray, int n, int p);
template <typename T> void load(std::string path, Matrice<T>& data_mat);

#include "loader_imp.h"

#endif