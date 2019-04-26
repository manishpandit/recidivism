# simple example of logistic regression on cleaned data
# balance using ai package and compare results
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegressionCV as LR
from sklearn.metrics import log_loss
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def min_max_scaling(X):
    scalar = MinMaxScaler(feature_range=(-1,1))
    scalar_fit = scalar.fit(X)
    dmin = scalar.data_min_
    dmax = scalar.data_max_
    return scalar

# load data
data_file = 'all-xy.csv'
Xy_df = pd.read_csv(data_file)

# prep train/dev datasets
Xy_train, Xy_test = train_test_split(Xy_df,test_size=0.2,random_state=42)
y_train = np.ndarray.astype(Xy_train.values[:,-1],int)
X_train = Xy_train.drop(Xy_train.columns[[0,-1]],axis=1)
y_test = np.ndarray.astype(Xy_test.values[:,-1],int)
X_test = Xy_test.drop(Xy_test.columns[[0,-1]],axis=1)
scalar = min_max_scaling(X_train)
X_train_norm = scalar.transform(X_train)
X_test_norm = scalar.transform(X_test)

# train initial model
clf = LR(penalty='l2', class_weight='balanced').fit(X_train_norm, y_train)
train_preds = clf.predict_proba(X_train_norm)[:,1]
test_preds = clf.predict_proba(X_test_norm)[:,1]
test_preds_label = clf.predict(X_test_norm)
class_preds = test_preds >= 0.5
train_ll = log_loss(y_train, train_preds)
test_ll = log_loss(y_test, test_preds)
print('test log-loss: ' + str(test_ll))

# working with ai360 -- first, run on initial model (no debiasing)
from aif360.datasets import StandardDataset
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

def fairness_metrics(classified_metric, log=False): # prints instead of return if log=True
    TPR = classified_metric.true_positive_rate()
    TNR = classified_metric.true_negative_rate()
    bal_acc = np.round(0.5*(TPR+TNR),2)
    dip = np.round((1/classified_metric.disparate_impact() - 1)*100,2)
    aod = np.round(classified_metric.average_odds_difference(),2)
    spd = np.round(classified_metric.statistical_parity_difference(),2)
    eod = np.round(classified_metric.equal_opportunity_difference(),2)
    if log:
        print("Balanced Accuracy: " + str(bal_acc))
        print("Disparate Impact (%): " + str(dip))
        print("Average Odds Difference: " + str(aod))
        print("Statistical Parity Difference: " + str(spd))
        print("Equal Opportunity Difference: " + str(eod))
    else:
        return bal_acc,dip,aod,spd,eod


privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]

comp_dataset = BinaryLabelDataset(
    df=Xy_test, favorable_label=0, unfavorable_label = 1,
    label_names = ['y'], protected_attribute_names = ['sex'],
    privileged_protected_attributes = [[0]],
    unprivileged_protected_attributes = [[1]]
)
comp_dataset_pred = comp_dataset.copy()
comp_dataset_pred.labels = test_preds_label
comp_dataset_pred.scores = test_preds

classified_metric = ClassificationMetric(
    comp_dataset, comp_dataset_pred, unprivileged_groups = unprivileged_groups,
    privileged_groups = privileged_groups)

# repeat with preprocessing
#from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
#from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from aif360.algorithms.preprocessing.reweighing import Reweighing

cd_train = BinaryLabelDataset(
    df=Xy_train, favorable_label=0, unfavorable_label = 1,
    label_names = ['y'], protected_attribute_names = ['sex'],
    privileged_protected_attributes = [[0]],
    unprivileged_protected_attributes = [[1]]
)
cd_test = BinaryLabelDataset(
    df=Xy_test, favorable_label=0, unfavorable_label = 1,
    label_names = ['y'], protected_attribute_names = ['sex'],
    privileged_protected_attributes = [[0]],
    unprivileged_protected_attributes = [[1]]
)

RW = Reweighing(unprivileged_groups=unprivileged_groups,
               privileged_groups=privileged_groups)
RW.fit(cd_train)
Xyt_train = RW.transform(cd_train)
Xyt_test = RW.transform(cd_test)

# prep train/dev datasets
yt_train = Xyt_train.labels.ravel()
Xt_train = Xyt_train.features
yt_test = Xyt_test.labels.ravel()
Xt_test = Xyt_test.features
scalar_fair = min_max_scaling(Xt_train)
Xt_train_norm = scalar_fair.transform(Xt_train)
Xt_test_norm = scalar_fair.transform(Xt_test)

# train initial model
clf = LR(penalty='l2', class_weight='balanced').fit(
    Xt_train_norm, yt_train, sample_weight=Xyt_train.instance_weights)
fair_train_preds = clf.predict_proba(Xt_train_norm)[:,1]
fair_test_preds = clf.predict_proba(Xt_test_norm)[:,1]
fair_test_preds_label = clf.predict(Xt_test_norm)
fair_class_preds = fair_test_preds >= 0.5
train_ll = log_loss(yt_train, fair_train_preds)
test_ll = log_loss(yt_test, fair_test_preds)
print('test log-loss: ' + str(test_ll))

# calculate fairness metrics for rebalanced model
fair_comp_dataset = BinaryLabelDataset(
    df=Xy_test, favorable_label=0, unfavorable_label = 1,
    label_names = ['y'], protected_attribute_names = ['sex'],
    privileged_protected_attributes = [[0]],
    unprivileged_protected_attributes = [[1]]
)
fair_comp_dataset_pred = fair_comp_dataset.copy()
fair_comp_dataset_pred.labels = fair_test_preds_label
fair_comp_dataset_pred.scores = fair_test_preds

fair_classified_metric = ClassificationMetric(
    fair_comp_dataset, fair_comp_dataset_pred, unprivileged_groups = unprivileged_groups,
    privileged_groups = privileged_groups)

# finally, compare!
fairness_metrics(classified_metric, log=True)
fairness_metrics(fair_classified_metric, log=True)
