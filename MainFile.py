#============================= IMPORT LIBRARIES =============================

import pandas as pd
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

#============================= DATA SELECTION ==============================

dataframe=pd.read_csv("Dataset.csv")

print("----------------------------------------------------")
print("Input Data          ")
print("----------------------------------------------------")
print()
print(dataframe.head(20))


#------ checking missing values --------

print("----------------------------------------------------")
print("              Handling Missing values               ")
print("----------------------------------------------------")
print()
print(dataframe.isnull().sum())

res = dataframe.isnull().sum().any()
        
if res == False:
    
    print("--------------------------------------------")
    print("  There is no Missing values in our dataset ")
    print("--------------------------------------------")
    print()    
    
        
else:

    print("--------------------------------------------")
    print(" Missing values is present in our dataset   ")
    print("--------------------------------------------")
    print()    
    
 
    
    dataframe = dataframe.fillna(0)
    
    resultt = dataframe.isnull().sum().any()
    
    if resultt == False:
        
        print("--------------------------------------------")
        print(" Data Cleaned !!!   ")
        print("--------------------------------------------")
        print()    
        print(dataframe.isnull().sum())



 # ================== DATA SPLITTING  ====================
 
from sklearn.model_selection import train_test_split    
    
X=dataframe.drop('Class',axis=1)

y=dataframe['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("---------------------------------------------")
print("             Data Splitting                  ")
print("---------------------------------------------")

print()

print("Total no of input data   :",dataframe.shape[0])
print("Total no of test data    :",X_test.shape[0])
print("Total no of train data   :",X_train.shape[0])



# # ================== HYBRID SAMPLING SMOTE + ENN  ====================

# from imblearn.under_sampling import EditedNearestNeighbours as ENN
# from imblearn.combine import SMOTEENN
# print("----------------------------------------------------")
# print("             Applying SMOTE-ENN                    ")
# print("----------------------------------------------------")

# smote_enn = SMOTEENN(sampling_strategy='auto', random_state=42)

# # Fit and transform the training data
# X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train[0:1000], y_train[0:1000])

# print("After applying SMOTE-ENN:")
# print("Number of training data after resampling: ", X_train_resampled.shape[0])
# print("Number of classes in y_train_resampled: ", pd.Series(y_train_resampled).value_counts())


# ================== HYBRID SAMPLING SMOTE + ENN  ====================

from imblearn.under_sampling import EditedNearestNeighbours as ENN
from imblearn.combine import SMOTEENN
import pandas as pd

print("----------------------------------------------------")
print("             Applying SMOTE-ENN                    ")
print("----------------------------------------------------")

# Initialize SMOTE-ENN with adjusted n_neighbors (set to 2 for this example)
smote_enn = SMOTEENN(sampling_strategy='auto', random_state=42, enn=ENN(n_neighbors=2))

# Fit and transform the training data (you can also try using the full training data: X_train and y_train)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

print("After applying SMOTE-ENN:")
print("Number of training data after resampling: ", X_train_resampled.shape[0])
print("Number of classes in y_train_resampled: ", pd.Series(y_train_resampled).value_counts())


# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import EditedNearestNeighbours as ENN
# from imblearn.combine import SMOTEENN
# import pandas as pd

# # First, apply SMOTE to balance classes
# smote = SMOTE(sampling_strategy='auto', random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# # Then apply ENN for noise cleaning
# smote_enn = SMOTEENN(sampling_strategy='auto', random_state=42, enn=ENN(n_neighbors=2))
# X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train_smote, y_train_smote)

# print("After applying SMOTE-ENN:")
# print("Number of training data after resampling: ", X_train_resampled.shape[0])
# print("Number of classes in y_train_resampled: ", pd.Series(y_train_resampled).value_counts())

# ================== CLASSIFCATION  ====================
 

# ------ RANDOM FOREST ------
 

 
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train,y_train)

pred_rf = rf.predict(X_test)


from sklearn import metrics

acc_rf = metrics.accuracy_score(pred_rf,y_test) * 100

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Cross-validation score
cv_scores = cross_val_score(rf, X_train, y_train, cv=5)  # Using 5-fold cross-validation

# Average of cross-validation scores
cv_avg_score = cv_scores.mean() * 100

# Confusion Matrix
conf_matrix_rf = confusion_matrix(y_test, pred_rf)


print("---------------------------------------------")
print("       Classification - Random Forest        ")
print("---------------------------------------------")

print()

print("1) Accuracy = ", acc_rf , '%')
print()
print("2) Classification Report")
print(metrics.classification_report(pred_rf,y_test))
print()
print("3) Error Rate = ", 100 - acc_rf, '%')
print()
print("4) Cross Validation Score = ", cv_avg_score,'%')
print()
print("5) Confusion Matrix ")
print()

import seaborn as sns
sns.heatmap(conf_matrix_rf, annot=True)
plt.title("Random Forest Classifier")
plt.show()



import pickle
with open('rf.pickle', 'wb') as f:
    pickle.dump(rf, f)


# 1 for fraudulent transactions, 0 otherwise



# ------ KNN ------
 

 
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X_train,y_train)

pred_knn = knn.predict(X_test)

from sklearn import metrics

acc_knn = metrics.accuracy_score(pred_knn,y_test) * 100


# Cross-validation score
cv_scores = cross_val_score(rf, X_train, y_train, cv=5)  # Using 5-fold cross-validation

# Average of cross-validation scores
cv_avg_score_knn = cv_scores.mean() * 100

# Confusion Matrix
conf_matrix_knn = confusion_matrix(y_test, pred_rf)



print("---------------------------------------------")
print("   Classification - K Nearest Neighbour      ")
print("---------------------------------------------")

print()

print("1) Accuracy = ", acc_knn , '%')
print()
print("2) Classification Report")
print(metrics.classification_report(pred_knn,y_test))
print()
print("3) Error Rate = ", 100 - acc_knn, '%')
print()
print("4) Cross Validation Score = ", cv_avg_score_knn,'%')
print()
print("5) Confusion Matrix ")
print()

import seaborn as sns
sns.heatmap(conf_matrix_knn, annot=True)
plt.title("KNN Classifier")
plt.show()


import pickle
with open('knn.pickle', 'wb') as f:
    pickle.dump(rf, f)


# ------ o	Hybrid RF + KNN ------

from sklearn.ensemble import VotingClassifier 
estimator = [] 
estimator.append(('RF',RandomForestClassifier())) 
estimator.append(('KNN', KNeighborsClassifier())) 
  
# Voting Classifier with hard voting 
vot_hard = VotingClassifier(estimators = estimator, voting ='soft') 
vot_hard.fit(X_train, y_train) 
y_pred_hyb = vot_hard.predict(X_test) 


acc_hyb = metrics.accuracy_score(y_pred_hyb,y_test) * 100


# Cross-validation score
cv_scores = cross_val_score(rf, X_train, y_train, cv=5)  # Using 5-fold cross-validation

# Average of cross-validation scores
cv_avg_score_hyb = cv_scores.mean() * 100

# Confusion Matrix
conf_matrix_hyb = confusion_matrix(y_test, pred_rf)



print("---------------------------------------------")
print("   Classification - Hybrid RF + KNN      ")
print("---------------------------------------------")

print()

print("1) Accuracy = ", acc_hyb , '%')
print()
print("2) Classification Report")
print(metrics.classification_report(y_pred_hyb,y_test))
print()
print("3) Error Rate = ", 100 - acc_hyb, '%')
print()
print("4) Cross Validation Score = ", cv_avg_score_hyb,'%')
print()
print("5) Confusion Matrix ")
print()

import seaborn as sns
sns.heatmap(conf_matrix_hyb, annot=True)
plt.title("Hybrid Classifier")
plt.show()


import pickle
with open('hyb.pickle', 'wb') as f:
    pickle.dump(rf, f)



# --- ROC CURVES


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Assuming you have your models and predictions
# For KNN
y_prob_knn = knn.predict_proba(X_test)[:, 1]  # Probabilities for class 1

# For Random Forest
y_prob_rf = rf.predict_proba(X_test)[:, 1]  # Probabilities for class 1

# For Hybrid model
y_prob_hyb = vot_hard.predict_proba(X_test)[:, 1]  # Probabilities for class 1

# Compute ROC curve and AUC for each model
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_hyb, tpr_hyb, _ = roc_curve(y_test, y_prob_hyb)

roc_auc_knn = auc(fpr_knn, tpr_knn)
roc_auc_rf = auc(fpr_rf, tpr_rf)
roc_auc_hyb = auc(fpr_hyb, tpr_hyb)

# Plotting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, color='blue', lw=2, label='KNN (AUC = %0.2f)' % roc_auc_knn)
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label='Random Forest (AUC = %0.2f)' % roc_auc_rf)
plt.plot(fpr_hyb, tpr_hyb, color='red', lw=2, label='Hybrid Model (AUC = %0.2f)' % roc_auc_hyb)

# Diagonal line (random classifier)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Add labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

# Show the plot
plt.show()
