from sklearn.model_selection import cross_val_score
import numpy as np
import time


def model_evaluation(clf, X, y):

    clf = clf

    t_start = time.time()
    clf = clf.fit(X, y)
    t_end = time.time()

    c_start = time.time()
    accuracy = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    f1_score = cross_val_score(clf, X, y, cv=5, scoring="f1_macro")
    roc_auc_ovr = cross_val_score(clf, X, y, cv=5, scoring="roc_auc_ovr")
    c_end = time.time()

    acc_mean_p = np.round(accuracy.mean() * 100, 2)
    f1_mean_p = np.round(f1_score.mean() * 100, 2)
    roc_auc_ovr_p = np.round(roc_auc_ovr.mean() * 100, 2)

    t_time = np.round((t_end - t_start), 3)
    c_time = np.round((c_end - c_start), 3)

    print(f"\n{clf}")
    print(f"The accuracy score of this classifier is:    {acc_mean_p}%.")
    print(f"The f1_macro score of this classifier is:    {f1_mean_p}%.")
    print(f"The roc_auc_ovr score of this classifier is: {roc_auc_ovr_p}%.")
    print(
        f"This classifier took {t_time} sec to train and {c_time} sec to evaluate CV and metric scores."
    )

    return accuracy.mean(), f1_score.mean(), roc_auc_ovr.mean()
