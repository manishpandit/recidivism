import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from aif360.datasets import StandardDataset, BinaryLabelDataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german

from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.metrics import ClassificationMetric

# calculates various fairness metrics
def fairness_metrics(classified_metric, log=False): # prints instead of return if log=True
    TPR = classified_metric.true_positive_rate()
    TNR = classified_metric.true_negative_rate()
    bal_acc = np.round(0.5*(TPR+TNR),4)
    dip = np.round((1/classified_metric.disparate_impact() - 1),4)
    aod = np.round(classified_metric.average_odds_difference(),4)
    spd = np.round(classified_metric.statistical_parity_difference(),4)
    eod = np.round(classified_metric.equal_opportunity_difference(),4)
    if log:
        print("Balanced Accuracy: " + str(bal_acc))
        print("Disparate Impact: " + str(dip))
        print("Average Odds Difference: " + str(aod))
        print("Statistical Parity Difference: " + str(spd))
        print("Equal Opportunity Difference: " + str(eod))
    else:
        return [bal_acc,dip,aod,spd,eod]

# redefine function since it's broken in aif360
def pr_predict(pr, dataset):
    data = np.column_stack([dataset.features, dataset.labels])
    columns = dataset.feature_names + dataset.label_names
    test_df = pd.DataFrame(data=data, columns=columns)
    all_sensitive_attributes = dataset.protected_attribute_names
    predictions, scores = pr._runTest(
        test_df, pr.class_attr, None,
        all_sensitive_attributes, pr.sensitive_attr, None)
    pred_dataset = dataset.copy()
    pred_dataset.labels = np.array([predictions]).T #predictions
    pred_dataset.scores = scores[:,1]
    return pred_dataset

def run_pr(eta, train_dataset, test_dataset):
    pr = PrejudiceRemover(eta=eta)
    pr.fit(train_dataset)
    train_preds = pr_predict(pr, train_dataset)
    test_preds = pr_predict(pr, test_dataset)
    ll_test = log_loss(test_dataset.labels, test_preds.scores)
    # analyze diversity metrics. race 2,0 is white,black to illustrate balanced metrics in one case
    privileged_groups = [{'race': [2]}]
    unprivileged_groups = [{'race': [0]}]
    train_metric = ClassificationMetric(
        train_dataset, train_preds, unprivileged_groups = unprivileged_groups,
        privileged_groups = privileged_groups)
    test_metric = ClassificationMetric(
        test_dataset, test_preds, unprivileged_groups = unprivileged_groups,
        privileged_groups = privileged_groups)
    return [ll_test] + fairness_metrics(test_metric)

# load data
data_file = 'all-xy.csv'
Xy_df = pd.read_csv(data_file)

# prep train/dev datasets
Xy_train, Xy_test = train_test_split(Xy_df,test_size=0.2,random_state=42)
# run prejudice remover algorithms
privileged_classes=np.where(Xy_train['race'] == 2)[0]
train_dataset = BinaryLabelDataset(
    df=Xy_train, favorable_label=0, unfavorable_label = 1,
    label_names = ['y'], protected_attribute_names = ['race'],
)
test_dataset = BinaryLabelDataset(
    df=Xy_test, favorable_label=0, unfavorable_label = 1,
    label_names = ['y'], protected_attribute_names = ['race']
)

# run prejudice remover
etas = [0,1,3,10,30,100,300,1000,3000]
metric_arr = np.zeros((len(etas),6))
for i in range(len(etas)):
    metric_arr[i,:] = run_pr(etas[i], train_dataset, test_dataset)

# plot results
import matplotlib.pyplot as plt

plt.plot(etas, metric_arr[:,1], label='Balanced Accuracy')
plt.plot(etas, metric_arr[:,2], label='Disparate Impact')
plt.plot(etas, metric_arr[:,3], label='Average Odds Difference')
plt.plot(etas, metric_arr[:,4], label='Statistical Parity Difference')
plt.plot(etas, metric_arr[:,5], label='Equal Opportunity Difference')
plt.xlabel('Eta (Bias Penalty)')
plt.ylabel('Metrics')
plt.legend()
plt.show()
