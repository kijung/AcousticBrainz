from __future__ import division
from script import *
import argparse
import random
import gc
from sklearn.preprocessing import normalize, StandardScaler, MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
import cPickle as pickle
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

def write(labels, keys, specific):
    with open(constants.path + specific + '_train_test.tsv', 'wb') as f:
        for n, key in enumerate(keys):
            f.writelines(('\t'.join([key] + list(labels[n])) + '\n').encode('utf-8'))

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
            files[line[0]] = set(line[2:])
    return files


def saveFeaturesTrain(train_files, test_files, specific, scalar, part):
    train = []
    test = []
    keys = list(train_files.keys())
    start = int(part) * len(keys)//5
    end = (int(part) + 1) * len(keys)//5
    count = 0
    used = []

    for f in keys[start:end]:
        path = constants.path + 'acousticbrainz-mediaeval-train/' + f[:2] + '/' + f + '.json'
        song = readjson(path)
        feat = getFeature(song)
        if len(feat) == 390:
             continue
        train.append(feat)
        used.append(train_files[f])
    print('Finished train ' + part)
    print(np.shape(train))

    path = constants.path + specific + '_mlb.pkl'
    with open(path, 'rb') as data_file:
        mlb = pickle.load(data_file)

    used = mlb.transform(used)
    data = dict()
    data['features'] = train
    data['genres'] = used
    with open(constants.path + specific + part + '_' + 'train.pkl', 'wb') as data_file:
        pickle.dump(data, data_file)
    train = 0
    data = 0
    used = 0
    gc.collect()
    
def saveFeaturesTest(train_files, test_files, specific, scalar, part):
    test = []
    keys = list(test_files.keys())
    start = int(part) * len(keys)//5
    end = (int(part) + 1) * len(keys)//5
    count = 0
    used = []
    mean = scalar.mean_
    print(len(mean))
    for f in keys[start:end]:
        path = constants.path + 'acousticbrainz-mediaeval-train/' + f[:2] + '/' + f + '.json'
        song = readjson(path)
        feat = getFeature(song)
        if len(feat) < 391:
            #dif = 391-len(feat)
            length = len(feat)
            print('Before :', len(feat))
            for m in mean[length:]:
                feat += [m]
            #print(mean[length:])
            print('After :', len(feat))
        test.append(feat)
        used.append(test_files[f])
    print('Finished test ' + part)
    print(np.shape(test))

    path = constants.path + specific + '_mlb.pkl'
    with open(path, 'rb') as data_file:
        mlb = pickle.load(data_file)

    test = scalar.transform(test)
    used = mlb.transform(used)
    data = dict()
    data['features'] = test
    data['genres'] = used
    with open(constants.path + specific + part + '_' + 'test.pkl', 'wb') as data_file:
        pickle.dump(data, data_file)
    test = 0
    data = 0
    used = 0
    gc.collect()

def train(classifier):
    xtrain = []
    ytrain = []
    print('Entering Classification')
    for n in range(5):
        print('Part ' + str(n))
        with open(constants.path + specific + str(n) + '_' + 'train.pkl', 'rb') as data_file:
            temp = pickle.load(data_file)
        print(np.shape(temp['features']), np.shape(temp['genres']))
        xtrain += temp['features']
        for m in temp['genres']:
            ytrain.append(m)
        temp = 0
        gc.collect()
    print(np.shape(xtrain), np.shape(ytrain))
    print(ytrain[0])   
    print("Let's train")
    #xtrain = xtrain[:10000]
    #ytrain = ytrain[:10000]
    gc.collect()
    classifier.fit(xtrain, ytrain)

    with open(constants.path + specific + '_classifier.pkl', 'wb') as data_file:
        pickle.dump(classifier, data_file)

def predict(classifier, part):
    with open(constants.path + specific + part + '_' + 'test.pkl', 'rb') as data_file:
        data = pickle.load(data_file)
    xtest = data['features']
    data = 0
    gc.collect()

    path = constants.path + specific + '_mlb.pkl'
    with open(path, 'rb') as data_file:
        mlb = pickle.load(data_file)

    ytest = mlb.inverse_transform(classifier.predict(xtest))
    return ytest

    #return classifier
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
    test_files = processTsv(test_file)
    """
    with open(constants.path + specific + '_scalar.txt', 'rb') as data_file:
        scalar = pickle.load(data_file)
    for n in range(5):
    	saveFeaturesTest(train_files, test_files, specific, scalar, str(n))
    
    print('Initialize Classifier')
    classifier = MultiOutputClassifier(LinearSVC(C=10, class_weight='balanced', dual=True), n_jobs = 4)
    print('Entering Train')
    train(classifier)
    
    with open(constants.path + specific + '_classifier.pkl', 'rb') as data_file:
        classifier = pickle.load(data_file)
    test_labels = []
    for n in range(5):
        ytest = predict(classifier, str(n))
        for m in ytest:
            test_labels.append(m)
        ytest = 0
        gc.collect()
    
    with open(constants.path + specific + '_predictions.pkl', 'wb') as data_file:
        pickle.dump(test_labels,data_file)  
    print(test_labels[0])
    """
    with open(constants.path + specific + '_predictions.pkl', 'rb') as data_file:
        test_labels = pickle.load(data_file)
    write(test_labels, list(test_files.keys()), specific)
