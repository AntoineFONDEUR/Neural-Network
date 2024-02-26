#ifndef MATRICE_H
#define MATRICE_H
#include <iostream>

template <class T> class Matrice;

template <class T> Matrice<T> cpc(Matrice<T> const &A, Matrice<T> const &B);
template <class T> Matrice<T> apply(T (*f)(T), Matrice<T> const &M);
template <class T> Matrice<T> operator*(Matrice<T> const &A, Matrice<T> const &B);
template <class T> Matrice<T> operator*(T lambda, Matrice<T> const&M);
template <class T> Matrice<T> operator+(Matrice<T> const &A, Matrice<T> const &B);
template <class T> Matrice<T> operator-(Matrice<T> const &A, Matrice<T> const &B);
template <class T> Matrice<T> operator-(Matrice<T> const &A, T x);
template <class T> Matrice<T> operator/(Matrice<T> const &A, T x);
template <class T> void operator<<(std::ostream &o, Matrice<T> const &matrice);

template <class T> class Matrice{
    public:
        int n;
        int p;
        int size;

        Matrice(int n=0, int p=0, bool random = false, double mean = 0, double variance = 1);
        Matrice(T* myarray, int n, int p);
        Matrice(Matrice<T> const &other);
        ~Matrice();
        void reshape(int n, int p);
        Matrice<T> transpose();
        float get(int i, int j) const;
        void set(int i, int j, T x);
        void apply_in_place(T (*f)(T));
        Matrice<T> subMatrice(int i_init, int j_init, int i_final, int j_final) const;
        void clip(T cilp_threshold_min, T cilp_threshold_max);
        T mean();
        T std();
        T max();
        T min();
        void describe();
        bool one_nan();
        bool one_inf();

        Matrice<T>& operator=(Matrice<T> const& other);
        friend Matrice<T> cpc<T>(Matrice<T> const &A, Matrice<T> const &B);
        friend Matrice<T> apply<T>(T (*f)(T), Matrice<T> const &M);
        friend Matrice<T> operator*<T>(Matrice<T> const &A, Matrice<T> const &B);
        friend Matrice<T> operator*<T>(T lambda, Matrice<T> const&M);
        friend Matrice<T> operator+<T>(Matrice<T> const &A, Matrice<T> const &B);
        friend Matrice<T> operator-<T>(Matrice<T> const &A, Matrice<T> const &B);
        friend Matrice<T> operator-<T>(Matrice<T> const &A, T x);
        friend Matrice<T> operator/<T>(Matrice<T> const &A, T x);
        friend void operator<<<T>(std::ostream &o, Matrice<T> const &matrice);

    private:
        T* array;

};

#include "matrice_imp.h"

#endif