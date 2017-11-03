from __future__ import division
from script import *
import argparse
import random
import gc
from sklearn.preprocessing import normalize, StandardScaler, MultiLabelBinarizer
import pickle
import constants
def writeToFile(data, path = 'results.json'):
    with open(path, 'w') as f:
        json.dump(data, f)
    #print(lowlevel_features())

def getLabels(genre, train_label, test_label, train_features):
    class_labels = []
    nonclass_labels = []
    test_labels = []
    class_features = []
    nonclass_features = []
    for n, label in enumerate(train_label):
        if genre in label:
            class_labels.append(1)
            class_features.append(train_features[n])
        else:
            nonclass_labels.append(0)
            nonclass_features.append(train_features[n])

    features = class_features
    sample_length = 0
    if len(class_labels) < 0.3 * len(nonclass_labels):
        sample_length = 2 * len(class_labels)
    else:
        sample_length = len(nonclass_labels)
    sample_length = min(len(nonclass_features), len(class_features))
    nonclass_features = random.sample(nonclass_features, sample_length)
    features += nonclass_features

    class_labels += [0 for n in range(sample_length)]
    features = list(zip(features, class_labels))
    random.shuffle(features)
    features, class_labels = zip(*features)
    #features = class_features + nonclass_features
    for label in test_label:
        if genre in label:
            test_labels.append(1)
        else:
            test_labels.append(0)

    return features, class_labels, test_labels

def pad_or_truncate(some_list, target_len):
    return some_list[:target_len] + [0]*(target_len - len(some_list))

def flatten(l): 
    return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]

def writeToTsv(genre_labels, subgenre_labels, keys):
    combine = []
    for n, key in enumerate(keys):
        detail = []
        detail.append(key)
        for genre in genre_labels:
            if genre_labels[genre][n] == 1:
                detail.append(genre)
                """
                for subgenre in subgenre_labels[genre]:
                    if subgenre_labels[genre][subgenre][n][0] < subgenre_labels[genre][subgenre][n][1]:
                        detail.append(subgenre)
                """
        combine.append(detail)
    with open('discogs_train_test_tonal.tsv', 'w') as f:
        for lst in combine:
            f.writelines(('\t'.join(lst) + '\n').encode('utf-8'))

def processTsv(tsv = 'acousticbrainz-mediaeval2017-discogs-train-train.tsv'):
    #a = # of entries, b = filter
    files = dict()
    with open(tsv) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        next(tsvreader, None)
        for line in tsvreader:
            while '' in line:
                line.remove('')
            #audio = Audio(line[0], line[1], line[2:])
            files[line[0]] = line[2:]
    return files


def saveFeatures(n, specific, scalar):
    path = constants.path + specific + str(n) + '_train.pkl'
    with open(path, 'rb') as data_file:
        data = pickle.load(data_file)
    features = data['features']

    features = scalar.transform(features)
    data['features'] = features

    with open(path, 'wb') as data_file:
        pickle.dump(data, data_file)

    #return scalar
    print('finished ' + str(n))
    #train[f]['rhythm'].pop('beats_loudness')
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script implements task 1 of the MediaEval 2017 challenge.")
    parser.add_argument('-i', '--input_file', required=True)
    #parser.add_argument('-mc', '--maingenre_classifier', default='linearsvc', help='The classifier to be used in the main genre. one of "rf", "mlp", "xgboost". Defaults to "mlp".')
    #parser.add_argument('-sc', '--subgenre_classifier', default='et', help='The classifier to be used in the sub genre. one of "rf", "mlp", "xgboost". Defaults to "mlp".')
    #parser.add_argument('-test', '--test_file', help='The pickled test file for the relevant dataset. If not provided, this script will use the train_test_split function of scikit.')
    #parser.add_argument('-m', '--model_file', help='The file the trained model should be written to.')
    #parser.add_argument('-o', '--output_file', required=True, help='The predicted classes will be written into this file, which then should be able to be evaluated with the R script provided by the challenge.')
    #parser.add_argument('-j', '--jobs', default=4, help='Number of parallel Jobs')
    args = parser.parse_args()
    #print(args)
    specific = args.input_file
    train_file = constants.path + 'acousticbrainz-mediaeval2017-' + specific + '-train-train.tsv'
    test_file = constants.path + 'acousticbrainz-mediaeval2017-' + specific + '-train-test.tsv'
    train_files = processTsv(train_file)

    train_labels = train_files.values()
    mlb = MultiLabelBinarizer()
    mlb.fit_transform(train_labels)

    path = constants.path + specific + '_train_labels.pkl'
    with open(path, 'wb') as data_file:
        pickle.dump(train_labels, data_file)

    path = constants.path + specific + '_mlb.pkl'
    with open(path, 'wb') as data_file:
        pickle.dump(mlb, data_file)

