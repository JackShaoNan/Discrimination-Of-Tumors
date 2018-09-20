# Discrimination-Of-Tumors
Discrimination of Malignant and Benign Liver Tumors Based on Machine Learning

1. Extracted ten first and second order texture features and labels from target volume of interest in tumor CT images

2. Used scikit-learn library to standardize datasets by using MinMaxScaler; ranked and selected features using PCA

3. Developed SVM and ANN models for training set, validated using ten-fold cross-validation method and proved that
SVM with radial basis function kernel has better generalization performance(auc=0.80)
