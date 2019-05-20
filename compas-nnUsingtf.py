
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from sklearn.model_selection import train_test_split

# working with ai360 -- first, run on initial model (no debiasing)
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

# hyper-parameters
epochs = 10
batch_size = 500
display_freq = 500
learning_rate = 0.001
alpha = 0.001   #need to find the best setting for with/without bias {with bias alpha = 0.1, without bias alpha = 0.001 - metric comparing disparate impact}
K = 1           #do we need to train adv every epoch or every K epoch

# data dimension
num_features = 15
n_classes = 1
n_bias = 1

h1 = 20    ## of nodes in 1st hidden layer in main NN
h2 = 20    ## of nodes in 2nd hidden layer in main NN

h3 = 20    ## of nodes in 1st hidden layer in adversary NN
h4 = 20    ## of nodes in 2nd hidden layer in adversary NN

def fairness_metrics(classified_metric, log=False): # prints instead of return if log=True
    TPR = classified_metric.true_positive_rate()
    TNR = classified_metric.true_negative_rate()
    bal_acc = np.round(0.5*(TPR+TNR),2)
    dip = np.round((1/classified_metric.disparate_impact() - 1)*100,2)
    aod = np.round(classified_metric.average_odds_difference(),2)
    spd = np.round(classified_metric.statistical_parity_difference(),2)
    eod = np.round(classified_metric.equal_opportunity_difference(),2)
    if log:
        print('True Positive Rate:' + str(np.round(TPR,2)))
        print('True Negative Rate:' + str(np.round(TNR,2)))
        print("Balanced Accuracy: " + str(bal_acc))
        print("Disparate Impact (%): " + str(dip))
        print("Average Odds Difference: " + str(aod))
        print("Statistical Parity Difference: " + str(spd))
        print("Equal Opportunity Difference: " + str(eod))
    else:
        return bal_acc,dip,aod,spd,eod

def min_max_scaling(X):
    scalar = MinMaxScaler(feature_range=(-10,10))
    scalar_fit = scalar.fit(X)
    dmin = scalar.data_min_
    dmax = scalar.data_max_
    return scalar

def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y

def get_next_batch(x, y, sex, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    sex_batch = sex[start:end]
    return x_batch, y_batch, sex_batch


# load data
data_file = 'all-xy-with-race.csv'
Xy_df = pd.read_csv(data_file)

# prep train/dev dataset files - need to do this only one time to maintain same train/validation set
Xy_train, Xy_valid = train_test_split(Xy_df,test_size=0.2,random_state=42)
Xy_train.to_csv('all-xy-with-race-train.csv')
Xy_valid.to_csv('all-xy-with-race-valid.csv')

# load train and validation data
train_data_file = 'all-xy-with-race-train.csv'
valid_data_file = 'all-xy-with-race-valid.csv'

Xy_train_df = pd.read_csv(train_data_file)
Xy_valid_df = pd.read_csv(valid_data_file)

y_train = np.ndarray.astype(Xy_train_df.values[:,-1],int)
y_train = np.reshape(y_train,(len(y_train),1))
x_train = Xy_train_df.drop(Xy_train_df.columns[[0,-1]],axis=1)
race_train = np.ndarray.astype(Xy_train_df.values[:,2],int)
race_train = np.reshape(race_train,(len(race_train),1))

y_valid = np.ndarray.astype(Xy_valid_df.values[:,-1],int)
y_valid = np.reshape(y_valid,(len(y_valid),1))
x_valid = Xy_valid_df.drop(Xy_valid_df.columns[[0,-1]],axis=1)
race_valid = np.ndarray.astype(Xy_valid_df.values[:,2],int)
race_valid = np.reshape(race_valid,(len(race_valid),1))

scalar = min_max_scaling(x_train)
x_train_norm = scalar.transform(x_train)
x_valid_norm = scalar.transform(x_valid)

# weight and bias wrappers
def weight_variable(name, shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W_' + name,dtype=tf.float32,shape=shape,initializer=initer)

def bias_variable(name, shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b_' + name,dtype=tf.float32,initializer=initial)

def fc_layer(x, num_units, name, use_relu=True, use_tanh=False):
    """
    Create a fully-connected layer
    :param x: input from previous layer
    :param num_units: number of hidden units in the fully-connected layer
    :param name: layer name
    :param use_relu: boolean to add ReLU non-linearity (or not)
    :return: The output array
    """
    in_dim = x.get_shape()[1]
    W = weight_variable(name, shape=[in_dim, num_units])
    b = bias_variable(name, [num_units])
    layer = tf.matmul(x, W)
    layer += b
    if use_relu:
        layer = tf.nn.relu(layer)
    if use_tanh:
        layer = tf.nn.tanh(layer)

    return layer

# Create the graph for the linear model
# Placeholders for inputs (x) and outputs(y)
x = tf.placeholder(tf.float32, shape=[None, num_features], name='X')
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')
xy = tf.placeholder(tf.float32, shape=[None, num_features+n_classes], name='XY')
race = tf.placeholder(tf.float32, shape=[None, n_classes], name='race')

# Create two fully-connected hidden layers of main NN
fc1 = fc_layer(x, h1, 'FC1', use_relu=False, use_tanh=True)
fc2 = fc_layer(fc1, h2, 'FC2', use_relu=False, use_tanh=True)
# Create a fully-connected layer with n_class nodes as output layer for main NN
output_logits = fc_layer(fc2, n_classes, 'OUT', use_relu=False, use_tanh=False)
# Network predictions for main NN
y_pred = tf.cast(output_logits >= 0.5, tf.float32)

# adversary network
fc3 = fc_layer(output_logits, h3, 'FC3', use_relu=False, use_tanh=True)
fc4 = fc_layer(fc3, h4, 'FC4', use_relu=False, use_tanh=True)
# Create a fully-connected layer for adversary network with bias class node as output layer
output_logits_adv = fc_layer(fc4, n_bias, 'ADV_OUT', use_relu=False, use_tanh=False) #here race is the bias class
# adversary network predictions
y_pred_adv = tf.cast(output_logits_adv >= 0.5, tf.float32)

# Define the loss function, optimizer, and accuracy
loss_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=race, logits=output_logits_adv), name='loss_adv')
optimizer_adv = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss_adv)
correct_prediction_adv = tf.equal(y_pred_adv, race, name='correct_pred_adv')
accuracy_adv = tf.reduce_mean(tf.cast(correct_prediction_adv, tf.float32), name='accuracy_adv')

# Define the loss function, optimizer, and accuracy
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output_logits), name='loss') - (alpha * loss_adv)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)
correct_prediction = tf.equal(y_pred, y, name='correct_pred')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

# Create the op for initializing all variables
init = tf.global_variables_initializer()


# Create an interactive session (to keep the session in the other cells)
sess = tf.InteractiveSession()
# Initialize all variables
sess.run(init)
# Number of training iterations in each epoch
num_tr_iter = int(len(y_train) / batch_size)
for epoch in range(epochs):
    print('Training epoch: {}'.format(epoch + 1))
    # Randomly shuffle the training data at the beginning of each epoch
    x_train_norm, y_train = randomize(x_train_norm, y_train)
    for iteration in range(num_tr_iter):
        start = iteration * batch_size
        end = (iteration + 1) * batch_size
        x_batch, y_batch, race_batch = get_next_batch(x_train_norm, y_train, race_train, start, end)
        # print('x=', x_batch, 'y =', y_batch)
        # print(x_batch)
        # Run optimization op (backprop)
        feed_dict_batch = {x: x_batch, y: y_batch, race: race_batch}
        sess.run(optimizer, feed_dict=feed_dict_batch)
        if epoch % K == 0:
            # print(epoch)
            sess.run(optimizer_adv, feed_dict=feed_dict_batch)

        # feed_adv_dict_batch = {x: output_logits, race: race_batch}
        # sess.run(optimizer, feed_dict=feed_adv_dict_batch)

        if iteration % display_freq == 0:
            # Calculate and display the batch loss and accuracy
            loss_batch, acc_batch, loss_adv_batch, acc_adv_batch = sess.run([loss, accuracy, loss_adv, accuracy_adv],feed_dict=feed_dict_batch)
            print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}, \tLoss_adv={1:.2f},\tTraining Adv Accuracy={2:.01%}".
                  format(iteration, loss_batch, acc_batch, loss_adv, accuracy_adv))

    # Run validation after every epoch
    feed_dict_valid = {x: x_valid_norm[:1443], y: y_valid[:1443], race: race_valid[:1443]}
    loss_valid, acc_valid, output_logits_valid, y_pred_valid, loss_adv_nn, acc_adv_nn = sess.run([loss, accuracy, output_logits, y_pred, loss_adv, accuracy_adv], feed_dict=feed_dict_valid)

    print('------------------------------------------------------------------------------------------------------------')
    print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
          format(epoch + 1, loss_valid, acc_valid))
    print("Epoch: {0}, adversary validation loss: {1:.2f}, adversary validation accuracy: {2:.01%}".
          format(epoch + 1, loss_adv_nn, acc_adv_nn))
    print('------------------------------------------------------------------------------------------------------------')

    privileged_groups = [{'race': 1}]
    unprivileged_groups = [{'race': 0}]

    comp_dataset = BinaryLabelDataset(
        df=Xy_valid_df, favorable_label=0, unfavorable_label = 1,
        label_names = ['y'], protected_attribute_names = ['race'],
        privileged_protected_attributes = [[0]],
        unprivileged_protected_attributes = [[1]]
    )
    comp_dataset_pred = comp_dataset.copy()
    comp_dataset_pred.labels = y_pred_valid
    comp_dataset_pred.scores = output_logits_valid

    classified_metric = ClassificationMetric(
        comp_dataset, comp_dataset_pred, unprivileged_groups = unprivileged_groups,
        privileged_groups = privileged_groups)
    fairness_metrics(classified_metric, log=True)