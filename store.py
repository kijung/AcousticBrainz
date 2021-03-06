from __future__ import division
from script import *
import argparse
import random
import gc
from sklearn.preprocessing import normalize, StandardScaler, MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_sample_weight
import cPickle as pickle
import constants
def dump(data, path):
    with open(path, 'wb') as data_file:
        pickle.dump(data, data_file)

def load(path):
    with open(path, 'rb') as data_file:
        return pickle.load(data_file)

def setup(train_files, test_files, specific):
    scalar = StandardScaler()
    mlb = MultiLabelBinarizer()
    train_labels = []
    train_data = []
    train_keys = []
    for f in train_files.keys():
        path = constants.path + 'acousticbrainz-mediaeval-train/' + f[:2] + '/' + f + '.json'
        song = readjson(path)
        feat = getFeature(song)
        if len(feat) != 391:
             continue
        train_keys.append(f)
        train_data.append(feat)
        train_labels.append(train_files[f]) 
    print('finished train')
    train_labels = mlb.fit_transform(train_labels)
    train_data = scalar.fit_transform(train_data)
    print('finished transforming')
    path = constants.path + specific + '_mlb.pkl'
    dump(mlb, path)

    path = constants.path + specific + '_scalar.pkl'
    dump(scalar, path)


    path = constants.path + specific + '_train.pkl'
    data = dict()
    data['features'] = train_data
    data['labels'] = train_labels
    data['keys'] = train_keys
    dump(data, path)

    print('finished dumping')
    #classifier = MultiOutputClassifier(LinearSVC(C=10, class_weight='balanced', dual=True), n_jobs = 4)
    classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=20, class_weight = 'balanced'), n_jobs=4)
    classifier.fit(train_data, train_labels)
    print('finished fitting')
    data = 0
    train_data = 0
    train_labels = 0
    train_keys = 0
    gc.collect()

    #test_labels = []
    test_data = []
    test_keys = list(test_files.keys())
    mean = scalar.mean_
    for f in test_keys:
        path = constants.path + 'acousticbrainz-mediaeval-train/' + f[:2] + '/' + f + '.json'
        song = readjson(path)
        feat = getFeature(song)   
        if len(feat) < 391:
            length = len(feat)
            for m in mean[length:]:
                feat += [m]             
        test_data.append(feat)
    test_data = scalar.transform(test_data)
    predictions = classifier.predict(test_data)
    print('finished predictions')
    genre_predictions = mlb.inverse_transform(predictions)
    write(genre_predictions, test_keys, specific)
    print('finished writing predictions')

def mycode(train_files, test_files, specific, indicies):
    indicies = np.array(indicies)
    #indicies = indicies[:len(indicies)//4]
    scalar = StandardScaler()
    mlb = MultiLabelBinarizer()
    train_labels = []
    train_data = []
    train_keys = []
    keys = list(train_files.keys())
    random.shuffle(keys)
    subset = 150000#len(keys)
    count = 0
    for f in keys[:subset]:
        count += 1
        path = constants.path + 'acousticbrainz-mediaeval-train/' + f[:2] + '/' + f + '.json'
        song = readjson(path)
        feat = getAllFeatures(song)
        if len(feat) != 2647:
            continue
        feat = np.array(feat)
        feat = feat[indicies]
        train_keys.append(f)
        train_data.append(feat)
        train_labels.append(train_files[f])
        if count % 10000 == 0:
            print("on ", count, "length of keys: ", len(train_keys))

    print('finished train')
    train_labels = mlb.fit_transform(train_labels)
    train_data = scalar.fit_transform(train_data)
    print('finished transforming')
    path = constants.path + specific + '_all2_mlb.pkl'
    dump(mlb, path)

    path = constants.path + specific + '_all2_scalar.pkl'
    dump(scalar, path)

    print(np.shape(train_data))
    path = constants.path + specific + '_all2_train.pkl'
    data = dict()
    data['features'] = train_data
    data['labels'] = train_labels
    data['keys'] = train_keys
    dump(data, path)

    print('finished dumping')
    #classifier = MultiOutputClassifier(LinearSVC(C=10, class_weight='balanced', dual=True), n_jobs = 4)
    classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=32, class_weight = 'balanced'), n_jobs=4)
    data = 0
    train_files = 0
    train_keys = 0
    keys = 0
    gc.collect()
    classifier.fit(train_data, train_labels)
    print('finished fitting')
    path = constants.path + specific + '_all2_classifier.pkl'
    dump(classifier, path)
    
    """
    with open(constants.path + specific + '_all2_scalar.pkl', 'rb') as data_file:
        scalar = pickle.load(data_file)
    with open(constants.path + specific + '_all2_mlb.pkl', 'rb') as data_file:
        mlb = pickle.load(data_file)
    with open(constants.path + specific + '_all2_classifier.pkl', 'rb') as data_file:
        classifier = pickle.load(data_file)
    """
    data = 0
    train_data = 0
    train_labels = 0
    train_keys = 0
    gc.collect()

    #test_labels = []
    test_data = []
    test_keys = list(test_files.keys())
    mean = scalar.mean_
    for f in test_keys:
        path = constants.path + 'acousticbrainz-mediaeval-train/' + f[:2] + '/' + f + '.json'
        song = readjson(path)
        feat = getAllFeatures(song)
        """
        if len(feat) < 2647:
            length = len(feat)
            for m in mean[length:]:
                feat += [m]
        """
        #feat = np.array(feat)
        if len(feat) < 2647:
            length = len(feat)
            print('Before: ', length)
            for m in range(2647 - length):
                feat += [np.random.rand()]
            #m = mean[indicies.index(2647)]
            #feat += [m]
            print('After: ', len(feat))
        feat = np.array(feat)
        feat = feat[indicies]             
        test_data.append(feat)
    test_data = scalar.transform(test_data)
    predictions = classifier.predict(test_data)
    print('finished predictions')
    genre_predictions = mlb.inverse_transform(predictions)
    write(genre_predictions, test_keys, specific)
    print('finished writing predictions')

def mycode2(classes, train_files, test_files, specific):
    for genre in classes:
        indicies = np.array(indicies)
        scalar = StandardScaler()
        mlb = MultiLabelBinarizer()
        train_labels = []
        train_data = []
        train_keys = []
        keys = list(train_files.keys())
        random.shuffle(keys)
        subset = len(keys)
        count = 0
        for f in keys[:subset]:
            count += 1
            path = constants.path + 'acousticbrainz-mediaeval-train/' + f[:2] + '/' + f + '.json'
            song = readjson(path)
            feat = getAllFeatures(song)
            if len(feat) != 2647:
                continue
            feat = np.array(feat)
            feat = feat[indicies]
            train_keys.append(f)
            train_data.append(feat)
            train_labels.append(train_files[f])
            if count % 10000 == 0:
                print("on ", count, "length of keys: ", len(train_keys))

        print('finished train')
        train_labels = mlb.fit_transform(train_labels)
        train_data = scalar.fit_transform(train_data)
        print('finished transforming')
        path = constants.path + specific + '_all2_mlb.pkl'
        dump(mlb, path)

        path = constants.path + specific + '_all2_scalar.pkl'
        dump(scalar, path)

        print(np.shape(train_data))
        path = constants.path + specific + '_all2_train.pkl'
        data = dict()
        data['features'] = train_data
        data['labels'] = train_labels
        data['keys'] = train_keys
        dump(data, path)

        print('finished dumping')
        #classifier = MultiOutputClassifier(LinearSVC(C=10, class_weight='balanced', dual=True), n_jobs = 4)
        classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=32, class_weight = 'balanced'), n_jobs=4)
        data = 0
        train_files = 0
        train_keys = 0
        keys = 0
        gc.collect()
        classifier.fit(train_data, train_labels)
        print('finished fitting')
        path = constants.path + specific + '_all2_classifier.pkl'
        dump(classifier, path)
        
        """
        with open(constants.path + specific + '_all2_scalar.pkl', 'rb') as data_file:
            scalar = pickle.load(data_file)
        with open(constants.path + specific + '_all2_mlb.pkl', 'rb') as data_file:
            mlb = pickle.load(data_file)
        with open(constants.path + specific + '_all2_classifier.pkl', 'rb') as data_file:
            classifier = pickle.load(data_file)
        """
        data = 0
        train_data = 0
        train_labels = 0
        train_keys = 0
        gc.collect()

        #test_labels = []
        test_data = []
        test_keys = list(test_files.keys())
        mean = scalar.mean_
        for f in test_keys:
            path = constants.path + 'acousticbrainz-mediaeval-train/' + f[:2] + '/' + f + '.json'
            song = readjson(path)
            feat = getAllFeatures(song)
            """
            if len(feat) < 2647:
                length = len(feat)
                for m in mean[length:]:
                    feat += [m]
            """
            #feat = np.array(feat)
            if len(feat) < 2647:
                length = len(feat)
                print('Before: ', length)
                for m in range(2647 - length):
                    feat += [np.random.rand()]
                #m = mean[indicies.index(2647)]
                #feat += [m]
                print('After: ', len(feat))
            feat = np.array(feat)
            feat = feat[indicies]             
            test_data.append(feat)
        test_data = scalar.transform(test_data)
        predictions = classifier.predict(test_data)
        print('finished predictions')
        genre_predictions = mlb.inverse_transform(predictions)
        write(genre_predictions, test_keys, specific)
        print('finished writing predictions')
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
            labels = []
            for m in line[2:]:
                if '---' not in m:
                    labels.append(m)
            files[line[0]] = set(labels)
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

"""
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
"""
"""
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
"""
def storeData(train_files, test_files, specific, m):
    #batches of 4?
    train_data = dict()
    train_data['features'] = []
    train_data['labels'] = []
    keys = list(train_files.keys())
    random.shuffle(keys)
    scalar = StandardScaler()
    
    #with open(constants.path + specific + '_scalar.pkl', 'rb') as data_file:
    #    scalar = pickle.load(data_file)
    with open(constants.path + specific + '_all_mlb.pkl', 'rb') as data_file:
        mlb = pickle.load(data_file)   
    for n in range(m):
        start = len(train_files)//m * n
        end = (n+1) * len(train_files)//m
        features = []
        labels = []
        for f in keys[start:end]:
            path = constants.path + 'acousticbrainz-mediaeval-train/' + f[:2] + '/' + f + '.json'
            song = readjson(path)
            feat = getAllFeatures(song)
            if len(feat) == 2647:
                features.append(feat)
                labels.append(train_files[f])
        scalar.partial_fit(features)
        labels = mlb.transform(labels)
        train_data['features'] = features
        train_data['labels'] = labels
        with open(constants.path + specific + '/train' + str(n) + '.pkl', 'wb') as data_file:
            pickle.dump(train_data, data_file)
        features = 0
        labels = 0
        train_data['features'] = []
        train_data['labels'] = []
        gc.collect()
    with open(constants.path + specific + '/scalar.pkl', 'wb') as data_file:
        pickle.dump(scalar, data_file)
    """
    keys = list(test_files.keys())
    for n in range(4):
        start = len(test_files)//4 * n
        end = (n+1) * len(test_files)//4
        features = []
        labels = []
        for f in keys[start:end]:
            path = constants.path + 'acousticbrainz-mediaeval-train/' + f[:2] + '/' + f + '.json'
            song = readjson(path)
            feat = getAllFeatures(song)
            if len(feat) == 2647:
                features.append(feat)
                labels.append(train_files[f])
        features = scalar.transform(features)
        labels = mlb.transform(labels)
        train_data['features'] = features
        train_data['labels'] = labels
        with open(constants.path + specific + '/test' + str(n) + '.pkl', 'wb') as data_file:
            pickle.dump(train_data, data_file)
        features = 0
        labels = 0
        train_data = 0
        gc.collect()   
    """
def scaleData(train_files, test_files,specific, m):
    with open(constants.path + specific + '/scalar.pkl', 'rb') as data_file:
        scalar = pickle.load(data_file)
    for n in range(m):
        with open(constants.path + specific + '/train' + str(n) + '.pkl', 'rb') as data_file:
            data = pickle.load(data_file)
        features = data['features']
        #labels = data['labels']
        features = scalar.transform(features)
        data['features'] = features
        with open(constants.path + specific + '/train' + str(n) + '.pkl', 'wb') as data_file:
            pickle.dump(data, data_file)
def trainData(specific, m):
    with open(constants.path + specific + '_all_mlb.pkl', 'rb') as data_file:
        mlb = pickle.load(data_file)
    for n in range(m):
        with open(constants.path + specific + '/train' + str(n) + '.pkl', 'rb') as data_file:
            data = pickle.load(data_file)
        features = data['features']
        labels = np.array(data['labels'])
        #weights = compute_sample_weight('balanced', labels)
        #classes = [np.unique(labels[:, i]) for i in range(labels.shape[1])]
        #classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=32, class_weight = 'balanced'), n_jobs=4)
        classifier = MultiOutputClassifier(LinearSVC(C=10, class_weight='balanced', dual=True), n_jobs=4)
        classifier.fit(features, labels) #classes = classes, sample_weight = weights)
        #data['features'] = features
        with open(constants.path + specific + '/classifier' + str(n) + '.pkl', 'wb') as data_file:
            pickle.dump(classifier, data_file)
def testData(test_files, specific, m):
    with open(constants.path + specific + '/scalar.pkl', 'rb') as data_file:
        scalar = pickle.load(data_file)

    with open(constants.path + specific + '_all_mlb.pkl', 'rb') as data_file:
        mlb = pickle.load(data_file)
    
    
    keys = list(test_files.keys())
    print(len(keys))
    test_length = len(keys)
    #test_data = dict()
    mean = scalar.mean_
    for n in range(m):
        test_data = dict()
        start = int((test_length)/m * n)
        end = int((test_length)/m * (n+1))
        print(start, end)
        features = []
        labels = []
        for f in keys[start:end]:
            path = constants.path + 'acousticbrainz-mediaeval-train/' + f[:2] + '/' + f + '.json'
            song = readjson(path)
            feat = getAllFeatures(song)
            #if len(feat) == 2647:
            if len(feat) < 2647:
                for a in range(len(feat), 2647):
                    feat.append(mean[a])
            features.append(feat)
            labels.append(f)
        print(n, np.shape(features))
        features = scalar.transform(features)
        #labels = mlb.transform(labels)
        test_data['features'] = features
        test_data['keys'] = labels
        with open(constants.path + specific + '/test' + str(n) + '.pkl', 'wb') as data_file:
            pickle.dump(test_data, data_file)
        #test_data = dict()
        #gc.collect()
def predictData(specific, m):
    with open(constants.path + specific + '_all_mlb.pkl', 'rb') as data_file:
        mlb = pickle.load(data_file)
    #with open(constants.path + specific + '/classifier.pkl', 'rb') as data_file:
    #    classifier = pickle.load(data_file)
    test_data = []
    test_keys = []
    for n in range(m):
        #with open(constants.path + specific + 
        with open(constants.path + specific + '/test' + str(n) + '.pkl', 'rb') as data_file:
            test = pickle.load(data_file)
        print(n, np.shape(test['features']))
        for m2 in test['features']:
            test_data.append(m2)
        for m2 in test['keys']:
            test_keys.append(m2)
        #test_keys += test['keys']
        print("AFTER ", np.shape(test_data))
    pred = np.zeros((len(test_keys), 31))
    #print(np.shape(test_data))
    for n in range(m):
        with open(constants.path + specific + '/classifier' + str(n) + '.pkl', 'rb') as data_file: 
            classifier = pickle.load(data_file)
        predictions = classifier.predict(test_data)
        pred += predictions
    #print(type(predictions), np.shape(pred))
    #print()
    with open(constants.path + specific + '/predictions.pkl','wb') as data_file:
        pickle.dump(pred, data_file)
    new_predictions = []
    for m in pred:
        new_predictions.append(m>1)
    #print(new_predictions[100])
    #print(new_predictions[101])
    #predictions = classifier.predict(test_data)
    #print('finished predictions')
    #genre_predictions = mlb.inverse_transform(predictions)
    #write(genre_predictions, test_keys, specific)
    #print('finished writing predictions')
    #return classifier
def predictionToLabels(specific, test_files):
    default = 'rock/pop'
    with open(constants.path + specific + '_all_mlb.pkl', 'rb') as data_file:
        mlb = pickle.load(data_file)

    with open(constants.path + specific + '/predictions.pkl','rb') as data_file:
        predictions = pickle.load(data_file)
    pred = []
    print((predictions[0] > 1).astype(float))
    for m in predictions:
        pred.append((m>1).astype(float))
    pred = np.array(pred)
    pred = mlb.inverse_transform(pred)
    print(pred[:10])
    predictions = []
    for m in pred:
        if len(m) == 0:
            predictions.append([default])
        else:
            p = []
            for n in m:
                p.append(n)
            predictions.append(p)
    print(predictions[20:30])
    print(len(predictions))
    print(len(test_files.keys()))
    write(predictions, list(test_files.keys()), specific)
def countLabels(specific, m):
    with open(constants.path + specific + '_all_mlb.pkl', 'rb') as data_file:
        mlb = pickle.load(data_file)
    count = 0
    mostPopular = 0
    genres = dict() 
    for n in range(m):
        with open(constants.path + specific + '/train' + str(n) + '.pkl', 'rb') as data_file:
            data = pickle.load(data_file)
        labels = data['labels']
        labels = mlb.inverse_transform(labels)
        for c in labels:
            for l in c:
                if l not in genres:
                    genres[l] = 0
                genres[l] += 1
                if count < genres[l]:
                    count = genres[l]
                    mostPopular = l
    with open(constants.path + specific + '/genres.pkl', 'wb') as data_file:
        pickle.dump(genres, data_file)
    print(mostPopular)
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
    path = constants.path + specific + '_indicies.pkl'
    with open(path, 'rb') as data_file:
        indicies = pickle.load(data_file)
    """
    m = 3
    storeData(train_files, test_files, specific, m)
    scaleData(train_files, test_files, specific, m)
    trainData(specific, m)
    testData(test_files, specific,m)
    predictData(specific, m)
    predictionToLabels(specific, test_files)
    #countLabels(specific, m)
    #indicies = np.arange(0, 2647)
    #mycode(train_files, test_files, specific, indicies)
    """
    path = constants.path + specific + '_all2_classifier.pkl'
    with open(path, 'rb') as data_file:
        classifier = pickle.load(data_file)
    #print(classifier.estimators_)
    path = constants.path + specific + '_all_mlb.pkl'
    with open(path, 'rb') as data_file:
        mlb = pickle.load(data_file)
    #print(mlb.classes_)
    #indicies = [0 for n in range(2647)]
    classes = dict()
    for m in mlb.classes_:
        classes[m] = 0
    for n, cl in enumerate(classifier.estimators_):
        importances = cl.feature_importances_
        s = importances
        importances = sorted(range(len(s)), key=lambda k: s[k], reverse=True)
        classes[n] = importances

    #s = indicies
    #indicies = sorted(range(len(s)), key=lambda k: s[k], reverse=True)
    path = constants.path + specific + '_genre_indicies.pkl'
    dump(classes, path)

    mycode2(classes, train_files, test_files, specific)
    #print(indicies[:20])
    #mycode(train_files, test_files, specific)
    """
    """
    with open(constants.path + specific + '_scalar.pkl', 'rb') as data_file:
        scalar = pickle.load(data_file)
    for n in range(5):
    	saveFeaturesTrain(train_files, test_files, specific, scalar, str(n))
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
    
    with open(constants.path + specific + '_predictions.pkl', 'rb') as data_file:
        test_labels = pickle.load(data_file)
    write(test_labels, list(test_files.keys()), specific)
    """
