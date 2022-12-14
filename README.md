# flare_forcasting_HHS_sciencefair_2022

## Background
Solar Flares are explosions of intense radiation coming from the release of magnetic energy associated with sunspots. 
Solar Flares can cause problems from communication disruptions to transformer explosions. 
Thus, it is necessary to predict these solar flares in order to prepare for the damage they will cause. 
Machine Learning is the use of computer algorithms that can learn from previous data to make inferences. 
This project will be using a Decision Tree Classifier, which is a supervised machine learning algorithm 
(models that model relationships between independent and dependent variable to predict new values based on different data). 
The classification algorithm will generate a subsequent Decision Tree to classify data. 
The decision tree consists of leaves which are the nodes that store Boolean statements the model uses to classify data. 
A Confusion Matrix will be used to measure the model’s performance. Figure 2 displays an example of a confusion matrix.
In datasets where the dependent variable is a true/false statement, then a confusion matrix can be helpful to get an overall view of the model’s performance.
The True Positive (TP) and True Negative (TN) values represent the correctly predicted false and true values.
I.e.) If a solar flare occurred or not, the False Positive (FP) and False Negative (FN) values represent the incorrectly predicted false and true values. 
From this confusion matrix, the F1-score can be derived. F1-score is a metric which provides a broader view of performance compared to accuracy.

## Methods
1. Import the training dataset and testing datasets; these will be the first 2 partitions of the dataset. ​ 
They will be referred to p1 and p2. This data will be taken from the Solar Weather Analytics for Solar Flares (SWAN-SF) dataset.
2. Clean and normalize the data; this will consist of removing the last two columns of the data and normalization 
(a scaling technique method in which data points are shifted and rescaled so that they end up in a range of 0 and 1). 
The type of standardization may differ based on different prototypes. A modified sklearn.preprocessing.normalize function was used to achieve this.
3. Undersample (the act of removing certain parts of the data to restore balance to the ratio of true and false dependent variables) the data 11 different times
– 10 times for p1 and once for p2.​ This will create 11 different datasets.
4. Separate the 11 under sampled dataframes from the independent variables and dependent variable(s). 
5. Use the sklearn.tree.DecisionTreeClassifer function to define 10 decision tree classifiers,
then fit the independent and dependent variables of the undersampled p1 dataset(s). This will train the model(s).
6. Generate predicted results by fitting the independent variables of the undersampled p2 dataset. This will result in 10 different predicted results. 
7. Generate 10 different F1-Scores based on the predicted results and the dependent variable of the under sampled p2 dataset.
Figure 3 is a flowchart that models how the machine learning model was trained.

