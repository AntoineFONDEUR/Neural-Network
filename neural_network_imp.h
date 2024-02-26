#include "neural_network.h"
#include <cmath>
#include <iostream>
#include <fstream>

template <typename T> NeuralNetwork<T>::NeuralNetwork(int input_size, char j_char){
    this->input_size = input_size;
    if (j_char == 'b'){
        this->L = &NeuralNetwork::BinaryCrossedEntropy;
        this->L_deriv = &NeuralNetwork::BinaryCrossedEntropy_deriv;
    }
    else if (j_char == 's'){
        this->L = &NeuralNetwork::SquaredMean;
        this->L_deriv = &NeuralNetwork::SquaredMean_deriv;
    }
}

template <typename T> NeuralNetwork<T>::~NeuralNetwork(){
    for (int i=0; i<top; i++){
        delete layers[i];
    }
}

template <typename T> void NeuralNetwork<T>::add_layer(int nodes, char func_char){
    int size_prev_layer = top == 0 ? input_size : layers[top-1]->nodes;
    layers[top] = new Layer<T>(nodes, size_prev_layer, func_char);
    top ++;
    nb_param+=nodes;
}

template <typename T> Matrice<T> NeuralNetwork<T>::predict(Matrice<T> const& X){
    T eps = 1e-5;
    Matrice<T> current_X = X;
    int m = X.n;
    for (int l=0; l<top; l++){
        // Calcul de Z : 
        Matrice<T> Z = Matrice<T>(m,layers[l]->nodes);
        for (int i=0; i<Z.n; i++){ 
            for (int j=0; j<Z.p; j++){
                for (int k=0; k<layers[l]->W.p; k++){
                    Z.set(i,j,Z.get(i,j)+layers[l]->W.get(j,k)*current_X.get(i,k));
                }
            }
        }

        //Calcul de la moyenne des colonnes de Z
        Matrice<T> moy = Matrice<T>(Z.p,1);
        for (int j=0; j<Z.p; j++){
            for (int i=0; i<Z.n; i++){
                moy.set(j,0,moy.get(j,0)+1.0/m*Z.get(i,j));
            }
        }
        //Calcul de la variance des colonnes de Z
        Matrice<T> var = Matrice<T>(Z.p,1);
        for (int j=0; j<Z.p; j++){
            for (int i=0; i<Z.n; i++){
                T ecart_moy = Z.get(i,j)-moy.get(j,0);
                var.set(j,0,var.get(j,0)+1.0/m*ecart_moy*ecart_moy);
            }
        }
        //Calcul de Z_chap
        Matrice<T> Z_chap = Matrice<T>(Z.n, Z.p);
        for (int i=0; i<Z_chap.n; i++){
            for (int j=0; j<Z_chap.p; j++){
                Z_chap.set(i,j,(Z.get(i,j)-moy.get(j,0))/std::sqrt(var.get(j,0)+eps));
            }
        }

        //Calcul de Z_BN ie Z "batch normed"
        Matrice<T> Z_BN = Matrice<T>(Z.n, Z.p);
        for (int i=0; i<Z_chap.n; i++){
            for (int j=0; j<Z_chap.p; j++){
                Z_BN.set(i,j,layers[l]->gamma.get(j,0)*Z_chap.get(i,j) + layers[l]->beta.get(j,0));
            }
        }

        //Mise à jour de l'input de la couche suivante
        current_X = apply<T>(layers[l]->activation_function, Z_BN);
    }
    return current_X;

}

template <typename T> void NeuralNetwork<T>::fit(Matrice<T> X, Matrice<T> Y, T alpha, int epochs, int m, T lambda, int verbose, bool log){
    double eps=1e-5;
    std::ofstream outputFile("log.txt", std::ios::out);
    if (log){
        outputFile << alpha << ',' << epochs << ',' << m << "," << lambda << std::endl;
    }
    for (int epoch = 0; epoch < epochs; epoch ++){
        for (int batch = 0; (batch+1)*m < X.n; batch++){
            //Forward prop : OK
            Matrice<T> X_batch = X.subMatrice(batch*m, 0, (batch+1)*m, X.p);
            Matrice<T> Y_real_batch = Y.subMatrice(batch*m, 0, (batch+1)*m, Y.p);
            Matrice<T> Y_pred_batch;
            layers[0]->X = X_batch;
            for (int l=0; l<top; l++){
                // Calcul de Z : 
                layers[l]->Z = Matrice<T>(m,layers[l]->nodes);
                for (int i=0; i<layers[l]->Z.n; i++){ 
                    for (int j=0; j<layers[l]->Z.p; j++){
                        for (int k=0; k<layers[l]->W.p; k++){
                            layers[l]->Z.set(i,j,layers[l]->Z.get(i,j)+layers[l]->W.get(j,k)*layers[l]->X.get(i,k));
                        }
                    }
                }

                //Calcul de la moyenne des colonnes de Z
                for (int j=0; j<layers[l]->Z.p; j++){
                    layers[l]->moy.set(j,0,0);
                    for (int i=0; i<layers[l]->Z.n; i++){
                        layers[l]->moy.set(j,0,layers[l]->moy.get(j,0)+1.0/m*layers[l]->Z.get(i,j));
                    }
                }
                //Calcul de la variance des colonnes de Z
                for (int j=0; j<layers[l]->Z.p; j++){
                    layers[l]->var.set(j,0,0);
                    for (int i=0; i<layers[l]->Z.n; i++){
                        T ecart_moy = layers[l]->Z.get(i,j)-layers[l]->moy.get(j,0);
                        layers[l]->var.set(j,0,layers[l]->var.get(j,0)+1.0/m*ecart_moy*ecart_moy);
                    }
                }
                //Calcul de Z_chap
                layers[l]->Z_chap = Matrice<T>(layers[l]->Z.n, layers[l]->Z.p);
                for (int i=0; i<layers[l]->Z_chap.n; i++){
                    for (int j=0; j<layers[l]->Z_chap.p; j++){
                        layers[l]->Z_chap.set(i,j,(layers[l]->Z.get(i,j)-layers[l]->moy.get(j,0))/std::sqrt(layers[l]->var.get(j,0)+eps));
                    }
                }
                if (verbose >2){
                    std::cout << "Forwardprop couche " << l << std::endl; 
                    std::cout << "\tZ_chap: ";
                    layers[l]->Z_chap.describe();
                }
                //Calcul de Z_BN ie Z "batch normed"
                layers[l]->Z_BN = Matrice<T>(layers[l]->Z.n, layers[l]->Z.p);
                for (int i=0; i<layers[l]->Z_chap.n; i++){
                    for (int j=0; j<layers[l]->Z_chap.p; j++){
                        layers[l]->Z_BN.set(i,j,layers[l]->gamma.get(j,0)*layers[l]->Z_chap.get(i,j) + layers[l]->beta.get(j,0));
                    }
                }
                if (verbose >2){
                    std::cout << "\tZ_BN: ";
                    layers[l]->Z_BN.describe();
                }
                //Mise à jour de l'input de la couche suivante
                if (l < top-1){layers[l+1]->X = apply<T>(layers[l]->activation_function, layers[l]->Z_BN);}
                else {Y_pred_batch = apply<T>(layers[l]->activation_function, layers[l]->Z_BN);}
            }
            
            //Backward prop :
            Matrice<T>* grad_W = new Matrice<T>[top];
            Matrice<T>* grad_gamma = new Matrice<T>[top];
            Matrice<T>* grad_beta = new Matrice<T>[top];
            Matrice<T> g_X = J_deriv(Y_pred_batch,Y_real_batch);
            if (verbose > 0){
                std::cout << "Epoch n°" << epoch+1 << "/" << epochs << ". Batch n°" << batch+1 << "/" << X.n/m << ". J = ";
                std::cout << J(Y_pred_batch,Y_real_batch,lambda) << ". J' = " << g_X.mean() << std::endl;
            }
            if (log){outputFile << J(Y_pred_batch,Y_real_batch,lambda) << std::endl;}

            for (int l=top-1; l>=0; l--){
                //Calcul du gradient de J par rapport à Z_BN
                if (verbose > 2){std::cout << "Retropropagation couche " << l << " :" << std::endl;}
                Matrice<T> g_Z_BN = cpc<T>(g_X,apply<T>(layers[l]->activation_func_deriv, layers[l]->Z_BN));
                if (verbose > 2){
                    std::cout << "\t g_Z_BN: ";
                    g_Z_BN.describe();
                }

                //Calcul du gradient de J par rapport à Z_chap
                Matrice<T> g_Z_chap = Matrice<T>(g_Z_BN.n, g_Z_BN.p);
                for (int i=0; i<g_Z_chap.n; i++){
                    for (int j=0; j<g_Z_chap.p; j++){
                        g_Z_chap.set(i,j,g_Z_BN.get(i,j)*layers[l]->gamma.get(j,0));
                    }
                }

                //Calcul du gradient de J par rapport à la moyenne et la variance:
                Matrice<T> g_var = Matrice<T>(g_Z_BN.p,1);
                for (int j=0; j<g_var.n; j++){
                    T diff = layers[l]->var.get(j,0) + eps;
                    for (int i=0; i<m; i++){
                        g_var.set(j,0,g_var.get(j,0) - 1.0/2 * g_Z_chap.get(i,j) * (layers[l]->Z.get(i,j) - layers[l]->moy.get(j,0)) / (std::sqrt(diff) * diff));
                    }
                }
                
                Matrice<T> g_moy = Matrice<T>(g_Z_BN.p,1);
                for (int j=0; j<g_moy.n; j++){
                    T diff = layers[l]->var.get(j,0) + eps;
                    for (int i=0; i<m; i++){
                        g_moy.set(j,0,g_moy.get(j,0) - g_Z_chap.get(i,j) * 1.0/std::sqrt(diff) - 2.0/m * g_var.get(j,0) * (layers[l]->Z.get(i,j) - layers[l]->moy.get(j,0)));
                    }
                }

                //Calcul du gradient de J par rapport à Z :
                Matrice<T> g_Z = Matrice<T>(g_Z_BN.n, g_Z_BN.p);
                for (int j=0; j<g_Z.p; j++){
                    for (int i=0; i<g_Z.n; i++){
                        g_Z.set(i,j, g_Z_chap.get(i,j) * 1.0/std::sqrt(layers[l]->var.get(j,0) + eps) + 2.0/m * g_var.get(j,0) * (layers[l]->Z.get(i,j) - layers[l]->moy.get(j,0)) + g_moy.get(j,0) * 1.0/m);
                    }
                }
                if (verbose > 2){
                    std::cout << "\t g_Z: ";
                    g_Z.describe();
                }

                //Calcul des gradients par rapport à W :
                grad_W[l] = Matrice<T>(layers[l]->W.n, layers[l]->W.p);
                for (int p=0; p<m; p++){
                    for (int i=0; i<layers[l]->W.n; i++){
                        for (int j=0; j<layers[l]->W.p; j++){
                            grad_W[l].set(i,j,grad_W[l].get(i,j) + 1.0/m * g_Z.get(p,i) * layers[l]->W.get(i,j) * layers[l]->X.get(p,j) + 1.0/(m*nb_param) * grad_W[l].get(i,j) * lambda);
                        }
                    }
                }
                if (verbose > 2){
                    std::cout << "\t grad_W: ";
                    grad_W[l].describe();
                }
                
                //Calcul des gradients par rapport à gamma et à bêta : OK
                grad_gamma[l] = Matrice<T>(layers[l]->gamma.n,layers[l]->gamma.p);
                grad_beta[l] = Matrice<T>(layers[l]->beta.n, layers[l]->beta.p);
                for (int i=0; i<layers[l]->gamma.n; i++){
                    for (int p=0; p<m; p++){
                        grad_gamma[l].set(i,0,grad_gamma[l].get(i,0) + g_Z_BN.get(p,i)*layers[l]->Z_chap.get(p,i));
                        grad_beta[l].set(i,0,grad_beta[l].get(i,0) + g_Z_BN.get(p,i));
                    }
                }
                if (verbose > 2){
                    std::cout << "\t grad_gamma: ";
                    grad_gamma[l].describe();
                    std::cout << "\t grad_beta: ";
                    grad_beta[l].describe();
                }

                //Mise à jour de g_X :
                if (l > 0){
                    Matrice<T> new_g_X = Matrice<T>(m,layers[l-1]->W.n);
                    for (int i=0; i<new_g_X.n; i++){
                        for (int j=0; j<new_g_X.p; j++){
                            for (int k=0; k<layers[l]->W.n; k++){
                                new_g_X.set(i,j,new_g_X.get(i,j) + g_Z.get(i,k) * layers[l]->W.get(k,j));
                            }
                        }
                    }
                    g_X = new_g_X;
                }
                if (verbose > 2){
                    std::cout << "\t g_X: ";
                    g_X.describe();
                }
            }

            for (int l=0; l<top; l++){
                if (verbose > 2){std::cout << "Update couche " << l << std::endl;}
                /*T cilp_threshold = 10;
                grad_W[l].clip(-cilp_threshold,cilp_threshold);
                grad_gamma[l].clip(-cilp_threshold,cilp_threshold);
                grad_beta[l].clip(-cilp_threshold,cilp_threshold);*/

                layers[l]->W = layers[l]->W - alpha * grad_W[l];
                layers[l]->gamma = layers[l]->gamma - alpha * grad_gamma[l];
                layers[l]->beta = layers[l]->beta - alpha * grad_beta[l];

                if (verbose > 2){
                    std::cout <<" \tW: ";
                    layers[l]->W.describe();
                    std::cout <<" \tgamma: ";
                    layers[l]->gamma.describe();
                    std::cout <<" \tbeta: ";
                    layers[l]->beta.describe();
                    //std::cout << std::endl;
                }   

                if (verbose == 2){std::cout << "Couche " << l << ". Grad W: " << grad_W[l].mean() << ". Grad gamma: " << grad_gamma[l].mean() << ". Grad beta: " << grad_beta[l].mean() << std::endl;}
            }
            if (verbose > 2){std::cout << std::endl;}

            delete[] grad_W;
            delete[] grad_gamma;
            delete[] grad_beta;
        }
    }
    outputFile.close();
}

template <typename T> T NeuralNetwork<T>::test(Matrice<T> const& X, Matrice<T> const& Y){
    float somme = 0;
    Matrice<T> Y_pred = predict(X);
    for (int i = 0; i < Y.n; i ++){
        int threshold = Y_pred.get(i,0) > 0.5 ? 1 : 0;
        //std::cout << i << ": " << "Expectations = " << Y_pred.get(i,0) << ", Reality = " << Y.get(i, 0) << std::endl;
        somme = Y.get(i, 0) == threshold ? somme + 1 : somme;
    }
    return somme / Y.n;
}

template <typename T> T NeuralNetwork<T>::J(Matrice<T> const& Y_chap, Matrice<T> const& Y, T lambda){
    T somme = 0;
    for (int i=0; i<Y_chap.n; i++){
        somme += (this->*L)(Y_chap.get(i,0), Y.get(i,0));
    }
    for (int l=0; l<top; l++){
        for (int i=0; i<layers[l]->W.n; i++){
            for (int j=0; j<layers[l]->W.p; j++){
                somme+=lambda/(2.0*nb_param) * layers[l]->W.get(i,j);
            }
        }
    }
    return 1.0/Y_chap.n * somme;
}

template <typename T> Matrice<T> NeuralNetwork<T>::J_deriv(Matrice<T> const& Y_chap, Matrice<T> const& Y){
    Matrice<T> somme(Y.n, Y.p);
    for (int i=0; i<Y_chap.n; i++){
        somme.set(i,0,(this->*L_deriv)(Y_chap.get(i,0), Y.get(i,0)));
    }
    return somme;
}

template <typename T> T NeuralNetwork<T>::SquaredMean(T y_chap, T y){
    return (y_chap - y)*(y_chap - y);
}

template <typename T> T NeuralNetwork<T>::SquaredMean_deriv(T y_chap, T y){
    return 2*(y_chap - y);
}

template <typename T> T NeuralNetwork<T>::BinaryCrossedEntropy(T y_chap, T y){
    return y == 1 ? - std::log(y_chap) : - std::log(1 - y_chap);
}

template <typename T> T NeuralNetwork<T>::BinaryCrossedEntropy_deriv(T y_chap, T y){
    T eps = 1e-5;
    return y == 1 ? - 1/(y_chap + eps) : 1/(1-y_chap + eps);
}