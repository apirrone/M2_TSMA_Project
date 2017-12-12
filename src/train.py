import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

# File for building and training the neural network

def one_hot_encode(labels):
    labels = labels.astype(int)

    for i in range(0, len(labels)):
        labels[i] = labels[i]-1
        
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    
    return one_hot_encode
    
def get_batch(inputX, inputY, batch_size):
    duration = len(inputX)

    c = list(zip(inputX, inputY))
    np.random.shuffle(c)
    inputX, inputY = zip(*c)
  
    for i in range(0,duration//batch_size):
        idx = i*batch_size
        yield inputX[idx:idx+batch_size], inputY[idx:idx+batch_size]


#Usage            
if len(sys.argv) != 5 :
    print("Usage : sudo python3 train.py <train_features_file.csv> <train_labels_file.csv> <test_features_file.csv> <test_labels_file.csv>")
    sys.exit()

train_features_file = sys.argv[1]
train_labels_file = sys.argv[2]

test_features_file = sys.argv[3]
test_labels_file = sys.argv[4]

# Load data
tr_features = np.loadtxt(open(train_features_file, "rb"), delimiter=",", dtype=np.str)
sc1 = StandardScaler().fit(tr_features)
tr_features = sc1.transform(tr_features);
tr_labels = np.loadtxt(open(train_labels_file, "rb"), delimiter=",", dtype=np.str)
tr_labels = one_hot_encode(tr_labels)

ts_features = np.loadtxt(open(test_features_file, "rb"), delimiter=",", dtype=np.str)
sc2 = StandardScaler().fit(ts_features)
ts_features = sc2.transform(ts_features);
ts_labels = np.loadtxt(open(test_labels_file, "rb"), delimiter=",", dtype=np.str)
ts_labels = one_hot_encode(ts_labels)

#Hyperparameters
training_epochs = 100000
n_dim = tr_features.shape[1]
n_classes = 8
n_hidden_units_one = 200
n_hidden_units_two = 300
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01
batch_size = 64


X = tf.placeholder(tf.float32,[None,n_dim], name="X")
Y = tf.placeholder(tf.float32,[None,n_classes], name="Y")

initializer = tf.contrib.layers.xavier_initializer()

# Initialize the weights with either normal distribution or xavier initializer

# W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd), name="W_1")
W_1 = tf.Variable(initializer([n_dim,n_hidden_units_one]),  name="W_1")
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd), name="b_1")
h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)

# Initialize the weights with either normal distribution or xavier initializer

# W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev=sd), name="W_2")
W_2 = tf.Variable(initializer([n_hidden_units_one,n_hidden_units_two]),  name="W_2")
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd), name="b_2")
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)

W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd), name="W" )
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd), name="b")

y_ = tf.nn.softmax(tf.matmul(h_2,W) + b, name="y_")

init = tf.global_variables_initializer()

#Either reduce mean, reduce sum of cross entropy 

#cost_function = -tf.reduce_sum(Y * tf.log(y_))
#cost_function = tf.reduce_mean(tf.pow(tf.subtract(y_, Y), 2))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# saver to save the state of the model during training (checkpoints)
saver = tf.train.Saver()

acc_history = np.empty(shape=[1],dtype=float)
y_true, y_pred = None, None

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        print("Epoch : "+str(epoch))
        avg_acc = 0.

        # training by batch
        for batch_x, batch_y in get_batch(tr_features, tr_labels, batch_size):
            _,acc = sess.run([optimizer,accuracy],feed_dict={X:batch_x,Y:batch_y})
            avg_acc += acc / batch_size

        print("Acc : "+str(avg_acc))
        print("-----")
        acc_history = np.append(acc_history,avg_acc)

        # Save model every 200 epoch
        if(epoch%200 == 0):
            saver.save(sess, "./model/test-acc_"+str(round(avg_acc, 2)), global_step=epoch)
            print("Test accuracy: ",round(sess.run(accuracy, feed_dict={X: ts_features,Y: ts_labels}),3))

        y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: ts_features})
        y_true = sess.run(tf.argmax(ts_labels,1))
        saver.save(sess, "./model/test_final")
        
    print("Test accuracy: ",round(sess.run(accuracy, feed_dict={X: ts_features,Y: ts_labels}),3))

fig = plt.figure(figsize=(10,8))
plt.plot(acc_history)
print(np.max(acc_history))
plt.axis([0,training_epochs,0,np.max(acc_history)])
plt.show()

p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average="micro")
print("F-Score:", round(f,3))
