# Comparison-of-Classifiers-for-Unsupervised-Anomaly-Detection

A comparison of 30 Unsupervised Anomaly Detection Classifieres among 90 Datasets.
- Author: Gabriel Ichcanziho Pérez Landa
- Date of creation: January 22th, 2021
- Email: ichcanziho@outlook.com


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
