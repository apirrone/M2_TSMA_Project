import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import re

# File for sampling from features arrays, needs features and trained model

np.set_printoptions(threshold=np.nan)

# Usage 
if len(sys.argv) != 6:
    print("USAGE : sudo python3 sample <model_file> <folder_files> <checkpoints_folder> <features> <ids>")
    sys.exit()

model = sys.argv[1]
path = sys.argv[2]
checkpoints_folder = sys.argv[3]
features = np.loadtxt(open(sys.argv[4], "rb"), delimiter=",", dtype=np.str)
sc = StandardScaler().fit(features)
features = sc.transform(features);
ids = np.loadtxt(open(sys.argv[5], "rb"), delimiter=",", dtype=np.str)

with tf.Session() as sess:
    
    # Restore variables from disk.
    saver = tf.train.import_meta_graph(model)
    saver.restore(sess, tf.train.latest_checkpoint(checkpoints_folder))
    print("Model restored")

    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("X:0")
    y_ = graph.get_tensor_by_name("y_:0")
    
    y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: features})

    print(re.sub(r' *\n *', '\n', np.array_str(np.c_[ids, y_pred+1]).replace('[', '').replace(']', '').replace(' *',', ').strip()))

    sys.exit()
