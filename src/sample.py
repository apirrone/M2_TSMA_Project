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

# File for sampling one or multiple files, needs audio file(s) and the trained model

np.set_printoptions(threshold=np.nan)

def scale(features, factor):

    for i in range(0, len(features)):
        for j in range(0, len(features[i])):
            val = float(features[i][j])
            val *= factor
            features[i][j] = str(val)

def extract_feature(file_name):
    
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    
    return mfccs,chroma,mel,contrast,tonnetz

def get_all_features(path_folder):
    features = np.empty((0, 193))
    ids = []
    
    file_ext = "*.mp3"
    n = 0
    for f in glob.glob(os.path.join(path_folder, file_ext)):
        try:
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(f)
        except Exception as e:
            print("Error encountered while parsing file: ", f)
            continue

        ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        features = np.vstack([features,ext_features])

        temp = f.split('/')
        temp = temp[len(temp)-1]
        id = temp.split('.')[0]
        ids = np.append(ids, id)
        
        n += 1

    return np.array(features), ids, n
    

#Usage
if len(sys.argv) < 4 or len(sys.argv) > 5:
    print("USAGE : sudo python3 sample -f (optional) <model_file> <file_to_classify.mp3> <checkpoints_folder>")
    sys.exit()
    
if len(sys.argv) == 5:
    if sys.argv[1] != "-f":
        print("USAGE : sudo python3 sample -f (optional) <model_file> <file_to_classify.mp3> <checkpoints_folder>")
        sys.exit()
    else:
        oneFile = True
        model = sys.argv[2]
        path = sys.argv[3]
        checkpoints_folder = sys.argv[4]
else:
    oneFile = False
    model = sys.argv[1]
    path = sys.argv[2]
    checkpoints_folder = sys.argv[3]


if oneFile:
    features = np.empty((0, 193))
    try:
        mfccs, chroma, mel, contrast,tonnetz = extract_feature(path)
    except Exception as e:
        print("Error encountered while parsing file: ", path)
        sys.exit()
        
    ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
    features = np.vstack([features,ext_features])
    features = np.array(features)
    temp = path.split('/')
    temp = temp[len(temp)-1]
    id = temp.split('.')[0]
    ids = [id]
    nb_files = 1
    
else:
    
    features, ids, nb_files = get_all_features(path)
    #Saving features
    np.savetxt("out_test/test_features.csv", features, delimiter=",")
    np.savetxt("out_test/test_ids.csv", np.array(ids).astype("str"), delimiter=",", fmt='%s')
    
    sc = StandardScaler().fit(features)
    features = sc.transform(features);

print("features extracted")

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
