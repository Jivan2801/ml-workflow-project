# Machine Learning Workflow Assignment

This repository presents a complete workflow of machine learning problems including supervised regression, dimensionality reduction, unsupervised clustering, and neural network-based classification and regression. The notebook demonstrates both theoretical understanding and practical implementation using real-world datasets.

---

## 📁 Files Included

- `Singh_Assignment_3.ipynb`: Jupyter Notebook containing full implementation
- `lab03_dataset_1.csv`: Dataset for regression problem
- `lab03_dataset_2.csv`: Dataset for PCA and SVD
- `lab03_dataset_3.csv`, `lab03_dataset_4.csv`: Datasets for clustering
- `lab03_dataset_5.csv`: Dataset for MLP classification
- `lab03_dataset_6.csv`: Dataset for MLP regression

---

## 📌 Problem 1: Regression Models

- Linear Regression, SGD Regressor, and KNN Regressor applied
- MSE computed for each model
- Data standardized using StandardScaler

---

## 📌 Problem 2: Dimensionality Reduction (PCA & SVD)

- Cleaned missing data
- Used PCA to find top k components that explain ≥90% variance
- Reconstructed projection using top PCs via SVD
- Extracted and compared eigenvalues from SVD

---

## 📌 Problem 3: Clustering Analysis

- Applied KMeans, GaussianMixture, Agglomerative, Spectral, and DBSCAN clustering
- Evaluated clusters for K=2 to K=6 using silhouette scores
- Visualized and compared cluster performance
- Identified DBSCAN as best for concentric rings structure

---

## 📌 Problem 4: MLP Classification

- Dataset: Airline satisfaction prediction
- Encoded and normalized inputs, used 70-30 split
- Built MLPClassifier with 3 hidden layers (ReLU and tanh)
- Compared MSE and training loss across activation functions

---

## 📌 Problem 5: MLP Regression

- Dataset: Gym/BMI data
- Encoded, normalized and split into train/test sets
- Compared MLPRegressor with `tanh` vs `sigmoid` activation
- Evaluated based on MSE

---

## 🔧 Requirements

```bash
pandas
numpy
scikit-learn
matplotlib
```

Install all dependencies using:

```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## 📷 Visuals & Evaluation

- Cluster plots across multiple algorithms
- Silhouette scores vs K curves
- Loss trajectory plots for MLP models
- PCA & SVD comparisons

---

## 📬 Author

**Jivan Singh**
