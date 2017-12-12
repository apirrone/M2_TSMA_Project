import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys

from matplotlib.pyplot import specgram

def extract_feature(file_name):
    
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    
    return mfccs,chroma,mel,contrast,tonnetz

def parse_files(ids_data, folder, out_directory, trainOrTest, file_ext=".mp3"):
    features, labels = np.empty((0, 193)), np.empty(0)
     
    nb = 0
    for id in ids_data:
        file_id = id[0]
        file_label = id[1]

        filepath = folder+str(file_id)+file_ext

        try:
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(filepath)
        except Exception as e:
            print("Error encountered while parsing file: ", filepath)
            continue
          
        ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        features = np.vstack([features,ext_features])
        
        labels = np.append(labels, file_label)
        nb += 1
        print(""+str(nb)+" files parsed")

        #save
        tmp_features = np.array(features)
        tmp_labels = np.array(labels, dtype=np.int)

        np.savetxt(""+out_directory+"/"+trainOrTest+"_features.csv", tmp_features, delimiter=",")
        np.savetxt(""+out_directory+"/"+trainOrTest+"_labels.csv", tmp_labels.astype(int), delimiter=",", fmt='%i')
        
    return np.array(features), np.array(labels, dtype=np.int)

# Select randomly 3/4 of the data for training, and the remaining 1/4 for testing (for cross-validation)
def extract_train_test(set_ids):
    np.random.shuffle(set_ids)
    sets = np.split(set_ids, [set_ids.size//8, set_ids.size-1])

    test = sets[0]
    train = sets[1]

    return train, test

# Usage
if len(sys.argv) != 4:
    print("Usage : sudo python3 extract.py <data_ids_file.csv> <data_folder> <output_directory>")
    sys.exit()


#Input
data_ids_file = sys.argv[1]
data_folder = sys.argv[2]
out_directory = sys.argv[3]


set_ids = np.loadtxt(open(data_ids_file, "rb"), delimiter=",", skiprows=1, dtype=np.str)

train_set_ids, test_set_ids = extract_train_test(set_ids)

np.savetxt(""+out_directory+"/train_set_ids.csv", train_set_ids.astype(int), delimiter=",", fmt='%i')
np.savetxt(""+out_directory+"/test_set_ids.csv", test_set_ids.astype(int), delimiter=",", fmt='%i')

train_features, train_labels = parse_files(train_set_ids, data_folder, out_directory, "train")
test_features, test_labels = parse_files(test_set_ids, data_folder, out_directory, "test")

np.savetxt(""+out_directory+"/train_features.csv", train_features, delimiter=",")
np.savetxt(""+out_directory+"/train_labels.csv", train_labels.astype(int), delimiter=",", fmt='%i')


