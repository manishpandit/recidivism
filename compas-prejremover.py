import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from aif360.datasets import StandardDataset, BinaryLabelDataset
from aif360.algorithms.inprocessing import PrejudiceRemover
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
        print("Disparate Impact (Neg): " + str(dip))
        print("Average Odds Difference: " + str(aod))
        print("Statistical Parity Difference: " + str(spd))
        print("Equal Opportunity Difference: " + str(eod))
    else:
        return np.array([bal_acc,-dip,aod,spd,eod])

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

def run_pr(eta, train_dataset, test_dataset, rs_dict):
    pr = PrejudiceRemover(eta=eta)
    pr.fit(train_dataset)
    test_preds = pr_predict(pr, test_dataset)
    return fairness_metrics_all(test_dataset, test_preds, rs_dict)

def fairness_metrics_all(test_dataset, test_preds, rs_dict):
    vals = rs_dict.values()
    fm_arr = np.zeros((len(vals),5))
    sensitives = np.array(test_dataset.protected_attributes)
    td_df = test_dataset.copy() ; tp_df = test_preds.copy()
    for val in vals:
        si = np.array([list(map(lambda x: int(x),sensitives == val))]).T # sensitive index
        td_df.protected_attributes = si ; tp_df.protected_attributes = si
        privileged_groups = [{'sensitive': 0}]
        unprivileged_groups = [{'sensitive': 1}]
        test_metric = ClassificationMetric(
            td_df, tp_df, unprivileged_groups = unprivileged_groups,
            privileged_groups = privileged_groups)
        fm_arr[val,:] = fairness_metrics(test_metric)
        print(fm_arr[val,:])
    return fm_arr

def build_race_sex_dict(dataset):
    rs_arr = np.unique(dataset[['race','sex']],axis=0)
    rs_tuples = list(map(lambda x: tuple(x), rs_arr))
    forced_others = [(1,1),(5,0),(5,1)]
    rs_dict = {} ; last_male_idx = len(rs_tuples) - len(forced_others) - 2
    idx = 0
    for rs_tuple in rs_tuples:
        if rs_tuple in forced_others: # assign to 'other' race (male/female)
            rs_dict[rs_tuple] = last_male_idx + rs_tuple[1]
        else:
            rs_dict[rs_tuple] = idx
            idx+=1
    return rs_dict

# mapping to race-sex tuples
rs_names = {
    0: 'African-American Male',
    1: 'African-American Female',
    2: 'Asian Male',
    3: 'Caucasian Male',
    4: 'Caucasian Female',
    5: 'Latino Male',
    6: 'Latino Female',
    7: 'Other Male',
    8: 'Other Female'
}

# load data
data_file = 'all-xy.csv'
Xy_df = pd.read_csv(data_file)
rs_dict = build_race_sex_dict(Xy_df) ; num_sensitive = len(set(rs_dict.values()))
Xy_df['sensitive'] = list(map(lambda x: rs_dict[tuple(x)], Xy_df[['race','sex']].values))
# show some stats (count,mean) of remaining classes
Xy_df.groupby('sensitive').count()[['y']] # count of each class
Xy_df.groupby('sensitive').mean()[['y']] # mean of each class
class_counts = np.array(Xy_df.groupby('sensitive').count()['y'])
class_weights = class_counts / np.sum(class_counts)

# prep train/dev datasets
Xy_train, Xy_test = train_test_split(Xy_df,test_size=0.2,random_state=142)
# run prejudice remover algorithms
train_dataset = BinaryLabelDataset(
    df=Xy_train, favorable_label=0, unfavorable_label = 1,
    label_names = ['y'], protected_attribute_names = ['sensitive'],
)
test_dataset = BinaryLabelDataset(
    df=Xy_test, favorable_label=0, unfavorable_label = 1,
    label_names = ['y'], protected_attribute_names = ['sensitive']
)

# run prejudice remover
etas = [0,1,5,10,20,50,100,125,150,175,200,250,500,1000]
metric_arr = np.zeros((len(etas),num_sensitive,5))
for i in range(len(etas)):
    metric_arr[i,:] = run_pr(etas[i], train_dataset, test_dataset, rs_dict)

# find best eta
metric_avg = np.zeros((len(etas),5))
for i in range(len(etas)):
    M_i = metric_arr[i]
    avg_metrics = np.dot(np.abs(M_i.T), class_weights)
    print(etas[i],avg_metrics)
    metric_avg[i,:] = avg_metrics

# plot results
import matplotlib.pyplot as plt

aam_arr = metric_arr[:,0,:] # all African-American male data

plt.plot(etas, aam_arr[:,0], label='Balanced Accuracy')
plt.plot(etas, aam_arr[:,1], label='Disparate Impact (-)')
plt.plot(etas, aam_arr[:,2], label='Average Odds Difference')
plt.plot(etas, aam_arr[:,3], label='Statistical Parity Difference')
plt.plot(etas, aam_arr[:,4], label='Equal Opportunity Difference')
plt.xlabel('Eta (Bias Penalty)')
plt.ylabel('Metrics (African-American Male)')
plt.legend()
plt.show()

plt.plot(etas, metric_avg[:,0], label='Balanced Accuracy')
plt.plot(etas, metric_avg[:,1], label='Disparate Impact (-)')
plt.plot(etas, metric_avg[:,2], label='Average Odds Difference')
plt.plot(etas, metric_avg[:,3], label='Statistical Parity Difference')
plt.plot(etas, metric_avg[:,4], label='Equal Opportunity Difference')
plt.xlabel('Eta (Bias Penalty)')
plt.ylabel('Metrics (Weighted Average, All Classes)')
plt.legend()
plt.show()
