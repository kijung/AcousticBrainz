from __future__ import division
from script import *
import random
import gc
from sklearn.preprocessing import normalize, StandardScaler
import pickle
from sklearn import linear_model
from sklearn.utils import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


def pad_or_truncate(some_list, target_len):
    return some_list[:target_len] + [0]*(target_len - len(some_list))

def flatten(l): return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]

def getGenres(files):
    stat = dict()
    files = processTsv(files)
    for f in files:
        gen = files[f]
        for g in gen:
            #if '---' in g:
            #    continue
            if g not in stat:
                stat[g] = 1
            else:
                stat[g] += 1
    genres = stat.keys()
    return stat

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


def rearrange(train_tsv, start, end, features, scalar):
    files = processTsv(train_tsv)
    lst = list(files.keys())
    random.shuffle(lst)
    lst = lst[start:end]
    #files = 0
    train = dict()
    length = dict()
    train_labels = []
    f = lst[0]
    path = './Downloads/acousticbrainz-mediaeval-train/' + f[:2] + '/' + f + '.json'
    with open(path) as data_file:
        train[f] = json.loads(data_file.read())
        train[f]['tonal'].pop('key_key')
        train[f]['tonal'].pop('key_scale')
        train[f]['tonal'].pop('chords_key')
        train[f]['tonal'].pop('chords_scale')
        train[f]['rhythm'].pop('beats_position')

    for f in lst:
        train_labels.append(files[f])
        path = './Downloads/acousticbrainz-mediaeval-train/' + f[:2] + '/' + f + '.json'
        if not os.path.isfile(path):
            continue
        with open(path) as data_file:
            train[f] = json.loads(data_file.read())
            train[f]['tonal'].pop('key_key')
            train[f]['tonal'].pop('key_scale')
            train[f]['tonal'].pop('chords_key')
            train[f]['tonal'].pop('chords_scale')
            train[f]['rhythm'].pop('beats_position')
            #train[f]['rhythm'].pop('beats_loudness')

    files = 0
    #scalar = dict()
    all_features = list(train[lst[0]]['tonal'].keys()) + list(train[lst[0]]['rhythm'].keys()) + list(train[lst[0]]['lowlevel'].keys())
    train_features = [[] for n in range(len(lst))]
    #indicies = ['' for n in range(3064)]
    index = 0
    for feature in all_features:
        length = len(scalar[feature].mean_)
        mat = []
        for key in lst:
            descriptor = 'lowlevel'
            if feature in train[key]['lowlevel']:
                descriptor = 'lowlevel'
            elif feature in train[key]['rhythm']:
                descriptor = 'rhythm'
            else:
                descriptor = 'tonal'
            c = train[key][descriptor][feature]
            if isinstance(c, dict):
                c = flatten(c.values())
            elif isinstance(c, float) or isinstance(c, int):
                c = [c]

            if len(c) < length:
                c += [np.mean(np.array(c)) for n in range(length - len(c))]
            if len(c) != length:
                print(len(c), length, feature)
            mat.append(c)
        mat = scalar[feature].transform(mat)
        #print(np.shape(mat))
        for n, key in enumerate(mat):
            train_features[n] += list(mat[n])
    #--------------test_features time-----------------#
    print('finished with train')
    #print(np.shape(train_features))
    return train_features, train_labels

def select(train_features, features):
    selective_features = [[] for n in range(len(train_features))]
    for i,n in enumerate(train_features):
        for m in features:
            selective_features[i] += [train_features[i][m]]
    print(np.shape(selective_features))
    return selective_features
def getLabels(genre, train_label, train_features):
    class_labels = []
    nonclass_labels = []
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
    sample_length = min(len(nonclass_features), sample_length)
    nonclass_features = random.sample(nonclass_features, sample_length)
    features += nonclass_features

    class_labels += [0 for n in range(sample_length)]
    features = list(zip(features, class_labels))
    random.shuffle(features)
    features, class_labels = zip(*features)
    #return train_features, train_labels
    return features, class_labels

if __name__ == "__main__":
    
    specific = 'discogs'
    train_file = 'acousticbrainz-mediaeval2017-' + specific + '-train-train.tsv'
    test_file = 'acousticbrainz-mediaeval2017-' + specific + '-train-test.tsv'

    genres = list(getGenres(train_file).keys())
    files = processTsv(train_file)
    train_length = len(files)
    files = 0

    with open(specific + '_weights.txt', 'rb') as data_file:
        weights = pickle.load(data_file)

    with open(specific + '_scalar.txt') as data_file:
        scalar = pickle.load(data_file)

    clf = dict()
    percent = [0.25, 0.50, 0.75]
    for genre in genres:
        clf[genre] = dict()
        for p in percent:
            clf[genre][p] = RandomForestClassifier(n_estimators = 64)

    features = []
    sample_size = 100000
    start = 0
    end = sample_size
    train_features, train_labels = rearrange(train_file, start, end, features, scalar)
    train_features, train_label = getLabels(genre, train_labels, train_features)


    for genre in genres:
        genre_importance = weights[genre]
        genre_importance.sort(key = lambda d: d[0], reverse=True)
        index = zip(*genre_importance)[1]
        size = len(index)
        index = list(index)
        #print(genre)

        for p in percent:
            size2 = int(size*p)
            features = index[:size2]
            select_features = select(train_features, features)
            clf[genre][p].fit(select_features, train_label, classes=np.array([0, 1]))
        joblib.dump(clf[genre], './' + specific + '/' + specific + "_randomforest_classifiers_" + genre + ".pkl")
        clf[genre] = 0
        gc.collect()




