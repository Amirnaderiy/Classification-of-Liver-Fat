# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 11:19:59 2022

@author: Amir Reza Naderi
"""

# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report

#reading dataset
path=r"C:\Users\Amir\Desktop\Paper3\selected_features.csv"
path2=r"C:\Users\Amir\Desktop\Paper3\targets.csv"
inputs = pd.read_csv(path, header=None)
targets = pd.read_csv(path2, header=None)


f1_score_svm=[]
accuracy_svm=[]
prec_svm=[]
recall_svm=[]

f1_lda=[]
accuracy_lda=[]
prec_lda=[]
recall_lda=[]

f1_score_KNN=[]
accuracy_KNN=[]
prec_KNN=[]
recall_KNN=[]

f1_score_LR=[]
accuracy_LR=[]
prec_LR=[]
recall_LR=[]

f1_score_RF=[]
accuracy_RF=[]
prec_RF=[]
recall_RF=[]

f1_score_XGB=[]
accuracy_XGB=[]
prec_XGB=[]
recall_XGB=[]

accuracy_dt=[]
prec_dt=[]
recall_dt=[]
f1_dt=[]

accuracy_GradientBoosting=[]
prec_GradientBoosting=[]
recall_GradientBoosting=[]
f1_GradientBoosting=[]

accuracy_voting=[]
prec_voting=[]
recall_voting=[]
f1_voting=[]


#classification
 
for i in range (100):
  X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.25)




  ##SVM
  svm_model = svm.SVC(kernel= "rbf",decision_function_shape='ovr')
  svm_model.fit(X_train, y_train)
  svm_pred=svm_model.predict(X_test)

  svm_prec=precision_score(y_test,svm_pred, average='macro')
  prec_svm.append (svm_prec)

  svm_recall=recall_score(y_test,svm_pred, average='macro')
  recall_svm.append (svm_recall)

  svm_f1= f1_score(y_test,svm_pred,average='macro')
  f1_score_svm.append(svm_f1)

  svm_score=accuracy_score(y_test, svm_pred)
  accuracy_svm.append(svm_score)
  
  
  
  ##Logistic Regression
  LR_class= LogisticRegression (random_state=0, max_iter=500).fit(X_train,y_train)
  LR_pred=LR_class.predict(X_test)

  LR_score=accuracy_score(y_test,LR_pred)
  accuracy_LR.append(LR_score)

  LR_pre=precision_score(y_test,LR_pred, average='macro')
  prec_LR.append(LR_pre)

  LR_recall=recall_score(y_test,LR_pred, average='macro')
  recall_LR.append(LR_recall) 

  LR_f1= f1_score(y_test,LR_pred,average='macro')
  f1_score_LR.append(LR_f1)



  ##KNN
  KNN_class= KNeighborsClassifier(n_neighbors = 5).fit(X_train,y_train)
  KNN_pred=KNN_class.predict(X_test)

  KNN_score=accuracy_score(y_test,KNN_pred)
  accuracy_KNN.append(KNN_score)

  KNN_pre=precision_score(y_test,KNN_pred, average='macro')
  prec_KNN.append(KNN_pre)

  KNN_recall=recall_score(y_test,KNN_pred, average='macro')
  recall_KNN.append(KNN_recall)

  KNN_f1= f1_score(y_test,KNN_pred,average='macro')
  f1_score_KNN.append(KNN_f1)



  #Random Forest
  random_forest = RandomForestClassifier(
    criterion='gini', 
    n_estimators=500,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=1,
    max_features='auto',
    oob_score=True,
    n_jobs=-1,
    verbose=1)
  random_forest.fit(X_train, y_train)
  RF_pred = random_forest.predict(X_test)

  RF_score=accuracy_score(y_test,RF_pred)
  accuracy_RF.append(RF_score)

  RF_pre=precision_score(y_test,RF_pred, average='macro')
  prec_RF.append(RF_pre)

  RF_recall=recall_score(y_test,RF_pred, average='macro')
  recall_RF.append(RF_recall)

  RF_f1= f1_score(y_test,RF_pred,average='macro')
  f1_score_RF.append(RF_f1)



  #XGBoost
  grid = {'max_depth':100}
  XGB_class = XGBClassifier()
  XGB_class.set_params(**grid)
  XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
        gamma=0, learning_rate=0.001, max_delta_step=0, max_depth=100,
        min_child_weight=1, missing=None, n_estimators=1000, nthread=-1,
        objective='binary:logistic', reg_alpha=0, reg_lambda=1,
        scale_pos_weight=1, seed=0, silent=True, subsample=1)
  XGB_class.fit(X_train, y_train)
  XGB_pred = XGB_class.predict(X_test)
  
  XGB_score=accuracy_score(y_test,XGB_pred)
  accuracy_XGB.append(XGB_score)

  XGB_pre=precision_score(y_test,XGB_pred, average='macro')
  prec_XGB.append(XGB_pre)

  XGB_recall=recall_score(y_test,XGB_pred, average='macro')
  recall_XGB.append(XGB_recall)
  
  XGB_f1= f1_score(y_test,XGB_pred,average='macro')
  f1_score_XGB.append(XGB_f1)
  
  ##Gradient Boosting
  GradientBoosting = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0,
                                                max_depth=1, random_state=0).fit(X_train, y_train)
  GradientBoosting_pred=GradientBoosting.predict(X_test)
  GradientBoosting_accuracy=accuracy_score(y_test,GradientBoosting_pred)
  accuracy_GradientBoosting.append(GradientBoosting_accuracy)
  GradientBoosting_prec= precision_score(y_test,GradientBoosting_pred,average='macro')
  prec_GradientBoosting.append(GradientBoosting_prec)
  GradientBoosting_recall= recall_score(y_test,GradientBoosting_pred,average='macro')
  recall_GradientBoosting.append(GradientBoosting_recall)
  GradientBoosting_f1= f1_score(y_test,GradientBoosting_pred,average='macro')
  f1_GradientBoosting.append(GradientBoosting_f1)
  
  
  #Linear Discriminant Analysis
  lda_model = LinearDiscriminantAnalysis()
  lda_model.fit(X_train, y_train)
  lda_pred=lda_model.predict(X_test)
  lda_score=accuracy_score(y_test,lda_pred)
  accuracy_lda.append(lda_score)
  lda_prec= precision_score(y_test,lda_pred,average='macro')
  prec_lda.append(lda_prec)
  lda_recall= recall_score(y_test,lda_pred,average='macro')
  recall_lda.append(lda_recall)
  lda_f1= f1_score(y_test,lda_pred,average='macro')
  f1_lda.append(lda_f1)

  ##Voting Classifier

  voting_model= VotingClassifier(estimators=[('lr',LR_class),('rf',random_forest),('svm',svm_model),('xgb',XGB_class),
                                  ('GB',GradientBoosting),('lda',lda_model)],
                                 weights=[1,1,1,1,1,1],flatten_transform=True)
  voting_model.fit(X_train, y_train)
  voting_pred=voting_model.predict(X_test)
  voting_score=accuracy_score(y_test,voting_pred)
  accuracy_voting.append(voting_score)
  voting_prec= precision_score(y_test,voting_pred,average='macro')
  prec_voting.append(voting_prec)
  voting_recall= recall_score(y_test,voting_pred,average='macro')
  recall_voting.append(voting_recall)
  voting_f1= f1_score(y_test,voting_pred,average='macro')
  f1_voting.append(voting_f1)

 
#print results
 
def print_metrics(name,accuracy,recall,f1_score,precision):
    print (f"The Accuracy of {name} is {np.mean(accuracy)*100:.2f} ± {np.std(accuracy)*100:.2f}")
    print (f"The Sensitivity of {name} is {np.mean(recall)*100:.2f} ± {np.std(recall)*100:.2f}")
    print (f"The F1-Score of {name} is {np.mean(f1_score)*100:.2f} ± {np.std(f1_score)*100:.2f}")
    print (f"The precision of {name} is {np.mean(precision)*100:.2f} ± {np.std(precision)*100:.2f}")
    print (10*"*****")

print_metrics("Logistic Regression",accuracy_LR,recall_LR,f1_score_LR,prec_LR)
print_metrics("SVM",accuracy_svm,recall_svm,f1_score_svm,prec_svm)
print_metrics("Random Forest",accuracy_RF,recall_RF,f1_score_RF,prec_RF)
print_metrics("XGBoost",accuracy_XGB,recall_XGB,f1_score_XGB,prec_XGB)
print_metrics("Gradient Boosting",accuracy_GradientBoosting,recall_GradientBoosting,f1_GradientBoosting,prec_GradientBoosting)
print_metrics("LDA",accuracy_lda,recall_lda,f1_lda,prec_lda)
print_metrics("Votting Classifier",accuracy_voting,voting_recall,f1_voting,prec_voting)


  


