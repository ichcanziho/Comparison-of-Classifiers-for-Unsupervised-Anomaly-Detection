# Comparison-of-Classifiers-for-Unsupervised-Anomaly-Detection

A comparison of 30 Unsupervised Anomaly Detection Classifieres among 90 Datasets.
- Code Author: [Gabriel Ichcanziho Pérez Landa](https://github.com/ichcanziho)
- Paper Authors: [Gabriel Ichcanziho Pérez Landa](https://www.linkedin.com/in/ichcanziho/), [Virginia Itzel Contreras Miranda](https://www.linkedin.com/in/itzel-contreras-5323abba/), [Daniela Macias Arregoyta](https://www.linkedin.com/in/daniela-macias-arregoyta/) and [Miguel Angel Medina-Pérez](https://sites.google.com/site/miguelmedinaperez/) 
- Date of creation: January 22th, 2021
- Code Author Email: ichcanziho@outlook.com

### Abstract

The problem of detecting anomalies in an unsuper-vised way is one of the most addressed in the fieldof machine learning. The anomaly detection algo-rithms are of the utmost importance, their field ofapplication is critical and ranges from cybersecu-rity to medicine. Finding these anomalies can bevery useful as it allows you to prevent problems.Despite the great relevance of this topic, few arti-cles compare different algorithms in depth. Thisdocument compares 30 anomaly detection algo-rithms using 90 publicly available databases. Theanomaly detection algorithms  belong to differ-ent families, being neural networks, probabilisticmodels, proximity-based, etc. The metrics usedin this paper are Area Under the Curve and Av-erage Precision, developing the analysis with noscaling, min-max scaling, and standard scalingthe databases. The results show that, with all thevariants, only six of the analyzed 30 algorithmshave the best performance, without statisticallysignificant differences among them.  A CriticalDifference diagram was created to show this com-parison. In the end, only three anomaly detectionalgorithms were determined as the best, outper-forming the rest. This research is useful for frauddetection, intrude detection, etc., and could beapplied to several fields of study.

### Installation

This repostory requires [Pip](https://docs.python.org/3/installing/index.html) to install the requirements.
Before install all the libraries, it is very recomendable to make a new [Virtual ENV](https://docs.python.org/3/library/venv.html) to isolate the new libraries.


To run the program you must have some libraries, you can install it using the next command:

```sh
$ pip install -r requirements.txt
```

## Implemented classifiers

- RandomForestClassifier()
- BRM()
- GaussianMixture()
- IsolationForest()
- OneClassSVM()
- EllipticEnvelope()
- KNN(method="mean")
- KNN(method="largest")
- KNN(method="median")
- PCA()
- COF()
- LODA()
- LOF()
- HBOS()
- MCD()
- FeatureBagging(combination='average')
- FeatureBagging(combination='max')
- CBLOF()
- FactorAnalysis()
- KernelDensity()
- COPOD()
- SOD()
- LSCP()
- LMDD(dis_measure='aad')
- LMDD(dis_measure='var')
- LMDD(dis_measure='iqr')
- SO_GAAL()
- MO_GAAL()
- VAE()
- AutoEncoder()
- OCKRA()

## Run the program


To generate the AUC and Average Precision results of the 30 models, run:

```sh
$ python main.py
```

To generate the Critical Diagram of AUC and Average Precision, run:

```sh
$ python make_cd_diagrams.py
```

To generate the Boxplot comparison, run:

```sh
$ python make_summarize_plot.py
```
