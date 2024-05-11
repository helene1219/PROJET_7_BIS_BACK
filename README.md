BACK
Au sein du projet P7 d'Openclassroom, ce dépôt est dédié aux tests et à la publication de l'api.
Une seule branche (principale)

Description
Dans ce dépôt, vous trouverez :

sous src : 
 - Les fichiers pickle 
 - api.py : le Script Python pour lancer une API et définissant les routes
 - assets : le script permettant la lecture des pickle
 - preprocess : le script reprenenant le préprocessing des data / modélisation

A à la racine
- le fichier main.py, est le fichier central d'appel aux fichiers "src"
- le fichier test_API.py définit les différents tests unitaires qui seront mis en oeuvre dans Github Actions.

Sous util : 
-  requirements.freeze reprend les versions des différentes librairies

Sous Github/workflow : 
- ci.yaml qui permet de faire des tests unitaires automatiques à chaque push ainsi que déploiement automatique

Pour lancer l’API en production :  https://scoring-p7.onrender.com/
