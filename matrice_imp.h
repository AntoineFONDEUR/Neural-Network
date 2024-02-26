#include "matrice.h"
#include <random>
#include <ctime>

template <typename T> Matrice<T>::Matrice(int n,int p,bool random, double mean, double variance){
    size = n*p;
    array = new T[size];

    //std::mt19937_64 rng(std::time(nullptr));
    //std::uniform_real_distribution<T> distribution(0.0, 1.0/12000);
    //double randomNumber = distribution(gen);

    std::random_device rd;
    std::mt19937 gen(rd());

    double standardDeviation = std::sqrt(variance);
    std::normal_distribution<T> distribution(mean, standardDeviation);
    //std::uniform_real_distribution<T> distribution(borne_inf, borne_sup);


    for (int i=0; i<n*p; i++){
        array[i] = random ? distribution(gen) : 0;
        //std::cout << array[i] << std::endl;
    }

    reshape(n,p);
}

template <typename T> Matrice<T>::Matrice(T* myarray, int n, int p){
    this->size = n*p;
    array = new T[size];
    for (int i=0; i<n*p; i++){
        array[i] = myarray[i];
    }
    reshape(n,p);

}

template <typename T> Matrice<T>::Matrice(Matrice const& other){
    this->n = other.n;
    this->p = other.p;
    this->size = other.size;
    this->array = new T[size];

    for (int i=0; i<this->size; i++){
        this->array[i] = other.array[i];
    }
}

template <typename T> Matrice<T>::~Matrice(){
    delete[] array;
}

template <typename T> void Matrice<T>::reshape(int n, int p){
    if (n*p != this->size){throw std::invalid_argument("Given n and p are not equal to size");}
    this->n = n;
    this->p = p;
}

template <typename T> Matrice<T> Matrice<T>::transpose(){
    Matrice<T> mat_transpose(this->p,this->n);
    for (int i=0; i<this->n; i++){
        for (int j=0; j<this->p; j++){
            mat_transpose.set(j, i, this->get(i,j));
        }
    }
    return mat_transpose;
}

template <typename T> float Matrice<T>::get(int i,int j) const{
    if (i >= n || j >= p){
        std::cout << "get(" << i << ',' << j << "). ";
        std::cout << "But matrix of size (" << this->n << ' '<< this->p << ')' << std::endl;
        throw std::out_of_range("Indexes are out of range");
    }
    return array[i*p+j];
}

template <typename T> void Matrice<T>::set(int i, int j, T x){
    if (i >= n || j >= p){
        std::cout << "set(" << i << ',' << j << "). ";
        std::cout << "But matrix of size (" << this->n << ' '<< this->p << ')' << std::endl;
        throw std::out_of_range("Indexes are out of range");
    }
    array[i*p + j] = x;
}

template <typename T> void Matrice<T>::apply_in_place(T (*f)(T)){
    for (int i=0; i<n; i++){
        for (int j=0; j<p; j++){
            set(i,j,f(get(i,j)));
        }
    }
}

template <typename T> void Matrice<T>::clip(T cilp_threshold_min, T cilp_threshold_max){
    for (int i=0; i<this->n; i++){
        for (int j=0; j<this->p; j++){
            T M_i_j = get(i,j);
            if (M_i_j < cilp_threshold_min){set(i,j,cilp_threshold_min);}
            if (M_i_j > cilp_threshold_max){set(i,j,cilp_threshold_max);}
            if (std::isnan(M_i_j)){set(i,j,cilp_threshold_max);}
        }
    }
}

template <typename T> T Matrice<T>::mean(){
    T my_mean = 0;
    for (int i=0; i<this->n; i++){
        for (int j=0; j<this->p; j++){
            my_mean += this->get(i,j);
        }
    }
    return my_mean/this->size;
}

template <typename T> bool Matrice<T>::one_nan(){
    for (int i=0; i<n; i++){
        for (int j=0; j<p; j++){
            if (std::isnan(get(i,j))){return true;}
        }
    }
    return false;
}

template <typename T> bool Matrice<T>::one_inf(){
    for (int i=0; i<n; i++){
        for (int j=0; j<p; j++){
            if (std::isinf(get(i,j))){return true;}
        }
    }
    return false;
}

template <typename T> T Matrice<T>::max(){
    T mymax = 0;
    for (int i=0; i<size; i++){
        if (std::isnan(array[i])){return array[i];}
        else if (std::isinf(array[i]) && array[i] > 0){return array[i];}
        else if (array[i] > mymax){mymax = array[i];}
    }
    return mymax;
}

template <typename T> T Matrice<T>::min(){
    T mymin = 0;
    for (int i=0; i<size; i++){
        if (std::isnan(array[i])){return array[i];}
        else if (std::isinf(array[i]) && array[i] < 0){return array[i];}
        else if (array[i] < mymin){mymin = array[i];}
    }
    return mymin;
}

template <typename T> T Matrice<T>::std(){
    T my_mean = this->mean();
    T my_std = 0;
    for (int i=0; i<this->n; i++){
        for (int j=0; j<this->p; j++){
            T diff = (this->get(i,j) - my_mean);
            my_std += (diff * diff);
        }
    }
    return std::sqrt(my_std/this->size);
}

template <typename T> void Matrice<T>::describe(){
    std::cout << "Min = " << min() << " Max = " << max() << " Mean = " << mean() << " Standard dev = " << std() << std::endl;
}

template <typename T> Matrice<T> Matrice<T>::subMatrice(int i_init, int j_init, int i_final, int j_final) const {
    Matrice<T> subM(i_final-i_init, j_final-j_init);
    for (int i=i_init; i<i_final; i++){
        for (int j=j_init; j<j_final; j++){
            subM.set(i-i_init, j-j_init, this->get(i,j));
        }
    }
    return subM;
}

template <typename T> Matrice<T>& Matrice<T>::operator=(Matrice<T> const& other){
    if (this != &other){
        this->n = other.n;
        this->p = other.p;
        this->size = other.size;

        delete[] array;
        array = new T[this->size];

        for (int i=0; i<this->size; i++){
            this->array[i] = other.array[i];
        }
    }
    return *this;
}

template <typename T> Matrice<T> operator*(Matrice<T> const &A, Matrice<T> const &B){
    if (A.p != B.n){
        std::cout << "Size of A: (" << A.n << ',' << A.p << "). Size of B: (" << B.n << ',' << B.p << ")" << std::endl;
        throw std::invalid_argument("Invalid matrix size for multiplication");}
    int shared_dim = B.n;
    Matrice<T> resultat(A.n, B.p);
    for (int i=0; i<resultat.n; i++){
        for (int j=0; j<resultat.p; j++){
            int sum=0;
            for (int k=0; k<shared_dim; k++){
                sum += A.get(i,k) * B.get(k,j);
            }
            resultat.set(i,j,sum);
        }
    }
    return resultat;
}

template <typename T> Matrice<T> operator+(Matrice<T> const &A, Matrice<T> const &B){
    if (A.n != B.n || A.p != B.p){throw std::invalid_argument("Matrix not of same size");}
    Matrice<T> resultat(A.n, A.p);
    for (int i=0; i<resultat.n; i++){
        for (int j=0; j<resultat.p; j++){
            resultat.set(i,j,A.get(i,j)+B.get(i,j));
        }
    }
    return resultat;
}

template <typename T> Matrice<T> operator*(T lambda,Matrice<T> const &M){
    Matrice<T> resultat(M.n, M.p);
    for (int i=0; i<resultat.n; i++){
        for (int j=0; j<resultat.p; j++){
            resultat.set(i,j,M.get(i,j)*lambda);
        }
    }
    return resultat;
}

template <typename T> Matrice<T> operator-(Matrice<T> const &A, Matrice<T> const &B){
    T moins_un = -1.0;
    return A + moins_un*B;
}

template <typename T> Matrice<T> operator-(Matrice<T> const &A, T x){
    Matrice<T> resultat(A.n, A.p);
    for (int i=0; i<resultat.n; i++){
        for (int j=0; j<resultat.p; j++){
            resultat.set(i,j,A.get(i,j)-x);
        }
    }
    return resultat;
}

template <typename T> Matrice<T> operator/(Matrice<T> const &A, T x){
    Matrice<T> resultat(A.n, A.p);
    for (int i=0; i<resultat.n; i++){
        for (int j=0; j<resultat.p; j++){
            resultat.set(i,j,A.get(i,j)/x);
        }
    }
    return resultat;
}

template <typename T> Matrice<T> cpc(Matrice<T> const &A, Matrice<T> const &B){
    if (A.n != B.n || A.p != B.p){throw std::invalid_argument("Matrix not of same size");}
    Matrice<T> mat_temp(A.n, A.p);
    for (int i=0; i<A.n; i++){
        for (int j=0; j<A.p; j++){
            mat_temp.set(i,j,A.get(i,j)*B.get(i,j));
        }
    }
    return mat_temp;
}

template <typename T> Matrice<T> apply(T (*f)(T), Matrice<T> const& M){
    Matrice<T> mat_temp = M;
    mat_temp.apply_in_place(f);
    return mat_temp;
}

template <typename T> void operator<<(std::ostream &o, Matrice<T> const &matrice){
        for (int i=0; i<matrice.n; i++){
            for (int j=0; j<matrice.p; j++){
                o << matrice.get(i,j);
                o << ' ';
            }
            if (i != matrice.n-1){o << std::endl;}
        }
        //o << std::endl;
}