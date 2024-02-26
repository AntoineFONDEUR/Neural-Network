# Neural-Network

Le code implémente les forwards et back propagation d'un résaeu de neurone avec des couches FC (pas de convolutions, résidus...).
Un batch norm est cependant intégré pour chaque couche.
L'exemple fourni fait tourner le programme sur le dataset de MNIST pour reconnaître des 0 et 1.
Sont recodés: une librairie pour le calcul matriciel et une librairie pour charger les datasets.
On voit dans le fichier plot la convergence de la fonction coût avec différents set d'hyperparamètres. Le nom des fichiers respecte le format: learning_rate - epochs - batch_size - coef_de_regularisation_l2.
