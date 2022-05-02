from sklearn.model_selection import cross_val_score
import numpy as np
import time

# function
def model_evaluation(clf, X, y):
    
    clf = clf 
    
    t_start = time.time()
    clf = clf.fit(X, y) 
    t_end = time.time() 
    
    c_start = time.time() 
    accuracy = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    f1_score = cross_val_score(clf, X, y, cv=10, scoring='f1_macro')
    roc_auc = cross_val_score(clf, X, y, cv=10, scoring='roc_auc')
    c_end = time.time() 
    
    acc_mean = np.round(accuracy.mean() * 100, 2)
    f1_mean = np.round(f1_score.mean() * 100, 2)
    
    t_time = np.round((t_end - t_start) / 60, 3) 
    c_time = np.round((c_end - c_start) / 60, 3)
    
    print(f'The accuracy score of this classifier is: {acc_mean}%.')
    print(f'The f1 score of this classifier is: {f1_mean}%.')
    print(f'The f1 score of this classifier is: {roc_auc}%.')
    print(f'This classifier took {t_time} minutes to train and {c_time} minutes to evaluate CV and metric scores.')