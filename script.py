import os
import sys
import re
import os.path
import ujson as json
import tensorflow as tf
import numpy as np
import pickle
import random
import csv
from classifier import *
from sklearn import svm, linear_model
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

class Audio(object):
    recordingmbid = ""
    releasegroupmbid = ""
    genres = []
    features = []

    # The class "constructor" - It's actually an initializer 
    def __init__(self, recordingid, groupid, genres):
        self.recordingmbid = recordingid
        self.releasegroupmbid = groupid
        self.genres = genres

    def __eq__(self, other):
        return self.recordingid == other.recordingid and self.releasegroupmbid == other.releasegroupmbid

    def inputFeatures(self, feature):
        self.features = feature

def readjson(path):
    with open(path) as data_file:
        data = json.load(data_file)
    return data

def lowlevel_features(path = './Downloads/acousticbrainz-mediaeval-train/08/0812194a-2575-4af5-812a-c00054137c7d.json'):
    with open(path) as data_file:
        data = json.loads(data_file.read())
    return list(data['lowlevel'].keys())

def show_feature(feature, path = './Downloads/acousticbrainz-mediaeval-train/08/0812194a-2575-4af5-812a-c00054137c7d.json'):
    with open(path) as data_file:
        data = json.loads(data_file.read())
    return data['lowlevel']['feature']

def writeToFile(data, path = 'results_discogs.json'):
    with open(path, 'w') as f:
        json.dump(data, f)

def readFeatures(path, genre = 'rock'):
    data = readjson(path)

    #name = []
    features = []
    labels = []
    for key in data.keys():
        features.append(data[key]['features'])
        if genre in data[key]['genres']:
            labels.append(1)
        else:
            labels.append(0)
    return list(data.keys()), np.array(features), np.array(labels)

def parseTsv(tsv = 'acousticbrainz-mediaeval2017-discogs-train-train.tsv'):
    #a = # of entries, b = filter
    files = dict()
    with open(tsv) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        next(tsvreader, None)
        for line in tsvreader:
            while '' in line:
                line.remove('')
            audio = Audio(line[0], line[1], line[2:])
            files[line[0]] = audio
    return files

def readTrain(files, descriptor = 'lowlevel', feature = 'mfcc', which = 'mean'):
    for f in files.keys():
        path = './Downloads/acousticbrainz-mediaeval-train/' + f[:2] + '/' + f + '.json'
        if not os.path.isfile(path):
            continue
        if which == '':
            with open(path) as data_file:
                files[f].inputFeatures(json.loads(data_file.read())[descriptor][feature])
            data_file = 0
        else:
            files[f].inputFeatures(read_json_file(path, descriptor, feature, which))
    return files

def stats(files):
    stat = dict()
    for f in files.keys():
        gen = files[f].genres
        for g in gen:
            if '---' in g:
                continue
            if g not in stat:
                stat[g] = 1
            else:
                stat[g] += 1
    return stat
        #now we do the dougie and hope for the best hehe xd
def pickleData(files, name='discogs_train_train_mean_mfcc.txt'):
    pickle.dump(files, open(name, "wb"))

def unpickleData(path = "discogs_train_train_mean_mfcc.txt"):
    data = pickle.load(open(path, "rb"))
    return data

def reformat(files, genre = 'rock'):
    #binary classifier
    features = []
    labels = []
    num_labels = 2
    #genre2 = genre + '/r'
    genre_keys = list(files.keys())
    random.shuffle(genre_keys)
    for f in genre_keys:
        features.append(files[f].features)
        if genre in files[f].genres:
            #features.append(files[f].mfcc)
            labels.append(1)
        else:
            labels.append(0)
    labels = np.array(labels)
    features = np.array(features)
    #labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return features, labels

def classify(train_features, train_labels, test_features, test_labels, genre = 'rock', classifier = 'SGD'):
    if classifier == 'SGD':
        return classifySGD(train_features, train_labels, test_features, test_labels, genre)
    elif classifier == 'RFC': #Random Forest Classifier
        return classifySklearn(train_features, train_labels, test_features, test_labels, genre, RandomForestClassifier(n_estimators = 64))
    elif classifier == 'SVM':
        return classifySklearn(train_features, train_labels, test_features, test_labels, genre, svm.SVC(probability = True))
    elif classifier == 'LR': #logistic regression
        return classifySklearn(train_features, train_labels, test_features, test_labels, genre, linear_model.LogisticRegression(C=1e5))
    elif classifier == 'kNR': #k nearest neighbors
        return classifySklearn(train_features, train_labels, test_features, test_labels, genre, knC = KNeighborsClassifier(13))
    else:
        return classifySklearn(train_features, train_labels, test_features, test_labels, genre, AdaBoostClassifier())

