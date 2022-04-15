# Participez à la conception d'une voiture autonome

Future Vision Transport est une entreprise qui conçoit des systèmes embarqués de vision par ordinateur pour les véhicules autonomes.

Vous êtes l’un des ingénieurs IA au sein de l’équipe R&D de cette entreprise. Votre équipe est composée d’ingénieurs aux profils variés. Chacun des membres de l’équipe est spécialisé sur une des parties du système embarqué de vision par ordinateur. 

Voici les différentes parties du système :
* 1. acquisition des images en temps réel
* 2. traitement des images
* 3. segmentation des images (c’est vous !)
* 4. système de décision

Vous travaillez sur la partie de segmentation des images (3) qui est alimentée par le bloc de traitement des images (2) et qui alimente le système de décision (4).

Votre rôle est de concevoir un premier modèle de segmentation d’images qui devra s’intégrer facilement dans la chaîne complète du système embarqué.

Lors d’une première phase de cadrage, vous avez récolté les avis de Franck et Laura, qui travaillent sur les parties avant et après votre intervention :
Franck, en charge du traitement des images (2) :
* Le jeu de données que Franck utilise est disponible à ce lien (https://www.cityscapes-dataset.com/dataset-overview/) (images segmentées et annotées de caméras embarquées). On a uniquement besoin des 8 catégories principales (et non pas des 32 sous-catégories)
* Les images en entrée peuvent changer. Le système d’acquisition n’est pas stable. 
* Le volume de données sera vite important.
Laura, en charge du système de décision (4)
* Souhaite une API simple à utiliser.
* L’API prend en entrée l’identifiant d’une image et renvoie la segmentation de l’image de l’algo, et de l’image réelle.

## Livrables 

* Les scripts développés sur Azure Machine Learning permettant l’exécution du pipeline complet 
  * Ce livrable vous servira à présenter le caractère “industrialisable” de votre travail en particulier le générateur de données.
* Une API Flask déployée grâce au service Azure qui recevra en entrée l’identifiant d’une image et retournera l’image avec les segments identifiés par votre modèle et l’image avec les segments identifiés annotés dans le jeu de données. Vous utiliserez les fonctionnalités d’Azure Machine Learning pour exposer facilement votre modèle entraîné  ;
  * Ce livrable permettra d’illustrer votre travail auprès de vos collègues et permettra à Laura d’utiliser facilement votre modèle.
* Une note technique de 10 pages environ contenant une présentation des différentes approches et une synthèse de l’état de l’art, la présentation plus détaillée du modèle et de l’architecture retenue, une synthèse des résultats obtenus (incluant les gains obtenus avec les approches d’augmentation des données) et une conclusion avec des pistes d’amélioration envisageables  :
  * Ce livrable vous servira à présenter votre démarche technique à vos collègues.
* Un support de présentation (type Power Point) de votre démarche méthodologique
  * Ce livrable vous permettra de présenter vos résultats à Laura.
