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
![image](https://user-images.githubusercontent.com/69658184/207741301-3b4c647c-7bf0-4bbc-9140-c4c444b8a2ac.png)


## Results
The machine learning model(s) had a mean F1-score of  83.327%.
Mean accuracy was 83%.
The Decision Tree generated had a depth of 22 and 104 leaves. Figure 4 displays the decision tree below, with a capped depth of 1 for viewing purposes. 
The data used in this classifier was scaled through log normalization.
Figures 5 and 6 visualize the model’s performance. The orange dot represents the model's performance (accuracy of F1-score). The color gradient represents the proficiency of any machine model. (A darker color corresponds to a higher performing model). Figures 4 & 5 are generated from GSU metrics.
![image](https://user-images.githubusercontent.com/69658184/207741235-e66f22ac-96a2-4094-8547-696475fc2686.png)
In Figures 5 & 6, the F1-score (F1), Accuracy, True Negative Rate(TNR), and True Positive Rate(TPR) are displayed are highlighted in teal below.
![image](https://user-images.githubusercontent.com/69658184/207741267-8f671a00-6c0a-4c79-a43c-35717a5fad52.png)

## Conclusion
The Machine Learning model had a desired performance being within the desirable range of 70.0-90.
The first iteration of the model was trained with 0-1 normalized data having a mean F1-score of 0.697. The current iteration was trained with log normalization. 
This yielded, a mean F1-score of  0.837. This change had a noticeable increase of 20.09%. This suggests that log normalization is better for boosting model performance.
In the future, modifying the parameters within the classifier may help the model more clearly comprehend the nuances in the data. 
Branching out from a Decision Tree Classifier to other classification models could also provide more insight into the nature of the SWAN-SF dataset, such as how different models process the data. A Support Vector Machine could be a promising choice for improving the F1-score.

