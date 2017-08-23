from __future__ import division
from script import *
import random
import gc
from sklearn.preprocessing import normalize, StandardScaler
import pickle
from sklearn import linear_model
from sklearn.utils import compute_class_weight

def writeToTsv(filename, genre_labels, subgenre_labels, keys):
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
    with open(filename, 'w') as f:
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

def pad_or_truncate(some_list, target_len):
    return some_list[:target_len] + [0]*(target_len - len(some_list))

def flatten(l): return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]

def getFeatures(test_tsv):
    files = processTsv(test_tsv)
    lst = list(files.keys())
    f = lst[300]
    path = './Downloads/acousticbrainz-mediaeval-train/' + f[:2] + '/' + f + '.json'
def extract(test_tsv, features, scalar, start, end):
    files = processTsv(test_tsv)
    lst = list(files.keys())
    print(len(lst), end-start)
    test_labels = []
    test = dict()
    test_features = [[] for n in range(end-start)]
    f = lst[0]
    path = './Downloads/acousticbrainz-mediaeval-train/' + f[:2] + '/' + f + '.json'
    lowlevel_features = 0
    train = dict()
    with open(path) as data_file:
        train[f] = json.loads(data_file.read())
        train[f]['tonal'].pop('key_key')
        train[f]['tonal'].pop('key_scale')
        train[f]['tonal'].pop('chords_key')
        train[f]['tonal'].pop('chords_scale')
        train[f]['rhythm'].pop('beats_position')
    features = list(train[lst[0]]['tonal'].keys()) + list(train[lst[0]]['rhythm'].keys()) + list(train[lst[0]]['lowlevel'].keys())
    lst = lst[start:end]
    for f in lst:
        test_labels.append(files[f])
        path = './Downloads/acousticbrainz-mediaeval-train/' + f[:2] + '/' + f + '.json'
        if not os.path.isfile(path):
            continue
        with open(path) as data_file:
            test[f] = json.loads(data_file.read())
            #del test[f]['tonal']
            #del test[f]['rhythm']
            del test[f]['tonal']['key_key']
            del test[f]['tonal']['key_scale']
            del test[f]['tonal']['chords_key']
            del test[f]['tonal']['chords_scale']
            del test[f]['rhythm']['beats_position']

            #for key in test[f]['lowlevel'].keys():
            #    if key not in features:
            #        del test[f]['lowlevel'][key]
    for feature in features:
        length = len(scalar[feature].mean_)
        mat = []
        for key in lst:
            descriptor = 'lowlevel'
            if feature in test[key]['tonal'].keys():
                descriptor = 'tonal'
            elif feature in test[key]['rhythm'].keys():
                descriptor = 'rhythm'
            c = test[key][descriptor][feature]
            if isinstance(c, dict):
                c = flatten(c.values())
            elif isinstance(c, float) or isinstance(c, int):
                c = [c]
            if len(c) < length:
                c += [np.mean(np.array(c)) for n in range(length - len(c))]
            mat.append(c)
        mat = scalar[feature].transform(mat)
        for n, key in enumerate(mat):
            test_features[n] += list(mat[n])
    return test_features, test_labels, lst

def select(train_features, features):
    selective_features = [[] for n in range(end-start)]
    for i,n in enumerate(train_features):
        for m in features:
            selective_features[i] += [train_features[i][m]]
    print(np.shape(selective_features))
    return selective_features

def getLabels(genre, test_label):
    #features = class_features + nonclass_features
    test_labels = []
    for label in test_label:
        if genre in label:
            test_labels.append(1)
        else:
            test_labels.append(0)

    return test_labels
if __name__ == "__main__":

    specific = 'discogs'
    train_file = 'acousticbrainz-mediaeval2017-' + specific + '-train-train.tsv'
    test_file = 'acousticbrainz-mediaeval2017-' + specific + '-train-test.tsv'

    features = ['spectral_complexity', 'spectral_contrast_coeffs', 'mfcc', 'gfcc', 'barkbands', 'spectral_contrast_valleys', 'pitch_salience', 'erbbands', 'melbands']
    
    genres = ['rock', 'pop', 'jazz', 'electronic']
    scalar = dict()
    clf = dict()
    with open('SGDClassifier.txt', 'rb') as data_file:
        clf = pickle.load(data_file)


    with open('indicies.txt', 'rb') as data_file:
        indicies = pickle.load(data_file)
    with open('weights.txt', 'rb') as data_file:
        weights = pickle.load(data_file)

    with open('SGDClassifier_scalar.tt') as data_file:
        scalar = pickle.load(data_file)

    with open('selective_weights.txt') as data_file:
        clf = pickle.load(data_file)

    test_prediction = dict()
    percent = [0.25, 0.50, 0.75]
    for p in percent:
        test_prediction[p] = dict()
        for genre in genres:
            test_prediction[p][genre] = []
    keys = []
    for n in range(4):
        start = n*60000
        end = (n+1)*60000
        if n ==3:
            end = 181404
        test_features, test_labels, subset_keys = extract(test_file, features, scalar, start, end)
        keys += subset_keys
        #print(np.shape(test_features))
        #test_prediction = dict()
        for genre in clf.keys():
            genre_importance = weights[genre]
            genre_importance.sort(key = lambda d: d[0], reverse=True)
            index = zip(*genre_importance)[1]
            size = len(index)
            index = list(index)
            test_label = getLabels(genre, test_labels)
            #test_label = test_label[n*30000: (n+1)*30000]
            for p in percent:
                size2 = int(size*p)
                features = index[:size2]
                #features.sort()
                #print(np.shape(test_features), np.shape(test_label))
                selective_features = select(test_features, features)
                print(genre, clf[genre][p].score(selective_features, test_label))
                test_prediction[p][genre] += list(clf[genre][p].predict(selective_features))

    for p in percent:
        writeToTsv(specific + '_' + str(p)+'_train_test_SGD.tsv', test_prediction[p], dict(), keys)
