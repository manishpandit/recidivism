# simple example of logistic regression on cleaned data
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
    return scalar.transform(X)

# load data
data_file = 'all-xy.csv'
Xy_df = pd.read_csv(data_file)
y = np.ndarray.astype(Xy_df.values[:,-1],int)
X_df = Xy_df.drop(Xy_df.columns[[0,-1]],axis=1)
X = np.ndarray.astype(X_df.values,float)

# train model
Xnorm = min_max_scaling(X)
X_train, X_test, y_train, y_test = train_test_split(Xnorm,y,test_size=0.2,random_state=42)
clf = LR(penalty='l2', class_weight='balanced').fit(X_train, y_train)
train_preds = clf.predict_proba(X_train)[:,1]
test_preds = clf.predict_proba(X_test)[:,1]
class_preds = test_preds >= 0.5
train_ll = log_loss(y_train, train_preds)
test_ll = log_loss(y_test, test_preds)
print('test log-loss: ' + str(test_ll))
#confusion_matrix(y_test, class_preds)
