from __future__ import division
from script import *
import random
import gc
from sklearn.preprocessing import normalize, StandardScaler
import pickle
from sklearn import linear_model
from sklearn.utils import compute_class_weight

#   files = parseTsv(tsv = 'acousticbrainz-mediaeval2017-discogs-train-train.tsv')
#files2 = parseTsv(tsv = 'acousticbrainz-mediaeval2017-discogs-train-test.tsv')


def show_feature(feature, path = './Downloads/acousticbrainz-mediaeval-train/08/0812194a-2575-4af5-812a-c00054137c7d.json'):
    with open(path) as data_file:
        data = json.loads(data_file.read())
    return data['lowlevel']['feature']

def writeToFile(data, path = 'results.json'):
    with open(path, 'w') as f:
        json.dump(data, f)
#print(lowlevel_features())
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

def getLabels(genre, train_label, train_features):
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

    return features, class_labels

def pad_or_truncate(some_list, target_len):
    return some_list[:target_len] + [0]*(target_len - len(some_list))

def flatten(l): return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]

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

def rearrange(train_tsv, test_tsv, start, end, features):
    files = processTsv(train_tsv)
    lst = list(files.keys())
    #random.shuffle(lst)
    lst = lst[start:end]
    #files = 0
    train = dict()
    length = dict()
    train_labels = []
    f = lst[0]

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

    files = 0
    scalar = dict()
    with open('discogs_scalar.txt', 'rb') as data_file:
        scalar = pickle.load(data_file)

    features = list(train[lst[0]]['tonal'].keys()) + list(train[lst[0]]['rhythm'].keys()) + list(train[lst[0]]['lowlevel'].keys())
    train_features = [[] for n in range(len(lst))]
    #indicies = ['' for n in range(3064)]
    index = 0
    for feature in features:
        mat = []
        length = len(scalar[feature].mean_)
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

            mat.append(c)
        """
        for n in range(length[feature]):
            indicies[n+index] = feature
        """
        #index += length[feature]
        #scalar[feature] = StandardScaler().fit(mat)
        mat = scalar[feature].transform(mat)
        #print(np.shape(mat))
        for n, key in enumerate(mat):
            train_features[n] += list(mat[n])
    #--------------test_features time-----------------#
    print('finished with train')
    print(np.shape(train_features))
    """
    files = processTsv(test_tsv)
    lst = list(files.keys())
    random.shuffle(lst)
    lst = lst[:test_subset_size]
    test = dict()
    test_labels = []
    #files = 0
    for f in lst:
        test_labels.append(files[f])
        path = './Downloads/acousticbrainz-mediaeval-train/' + f[:2] + '/' + f + '.json'
        if not os.path.isfile(path):
            continue
        with open(path) as data_file:
            test[f] = json.loads(data_file.read())
            test[f]['tonal'].pop('key_key')
            test[f]['tonal'].pop('key_scale')
            test[f]['tonal'].pop('chords_key')
            test[f]['tonal'].pop('chords_scale')
            test[f]['rhythm'].pop('beats_position')
            #test[f]['rhythm'].pop('beats_loudness')
    files = 0
    test_features = [[] for n in range(len(lst))]
    for feature in features:
        mat = []
        for key in lst:
            descriptor = 'lowlevel'
            
            if feature in test[key]['lowlevel']:
                descriptor = 'lowlevel'
            elif feature in test[key]['rhythm']:
                descriptor = 'rhythm'
            else:
                descriptor = 'tonal'
            
            c = test[key][descriptor][feature]
            if isinstance(c, dict):
                c = flatten(c.values())
            elif isinstance(c, float) or isinstance(c, int):
                c = [c]
            if len(c) < length[feature]:
                c += [np.mean(np.array(c)) for n in range(length[feature] - len(c))]
            mat.append(c)
        #scalar[feature] = StandardScaler().fit(mat)
        mat = scalar[feature].transform(mat)
        for n, key in enumerate(mat):
            test_features[n] += list(mat[n])
    print(np.shape(test_features))
    return train_features, train_labels, test_features, test_labels, lst
    """
    return train_features, train_labels

if __name__ == "__main__":
    #data = readjson('discogs_train_train_rhythm.json')
    #data = readjson('discogs_train_train_tonal.json')
    #genres = getGenres(data).keys()
    #genres.sort()
    #subgenres = []
    scalar = dict()
    clf = dict()
    with open('SGDClassifier.txt', 'rb') as data_file:
        clf = pickle.load(data_file)

    with open('SGDClassifier_scalar.tt') as data_file:
        scalar = pickle.load(data_file)

    
    specific = 'discogs'
    train_file = 'acousticbrainz-mediaeval2017-' + specific + '-train-train.tsv'
    test_file = 'acousticbrainz-mediaeval2017-' + specific + '-train-test.tsv'
    genres = getGenres(train_file).keys()
    genres.sort()
    main_genres = dict()
    for gen in genres:
        if '---' in gen:
            main_genres[gen.split('---')[0]].append(gen)
        else:
            main_genres[gen] = []

    #genres = ['rock', 'pop', 'jazz', 'electronic']
    data = 0
    clf = dict()
    for genre in main_genres.keys():
        clf[genre] = linear_model.SGDClassifier()
        for g in main_genres[genre]:
            clf[g] = linear_model.SGDClassifier()

    #train_features, train_labels, test_features, test_labels, test_keys = rearrange(train_file, test_file, 100000, 80000, features)
    genre_labels = dict()
    subgenre_labels = dict()
    train_length = len(list(processTsv(train_file).keys()))
    subset_length = 80000
    iterations = train_length//subset_length + 1
    for m in range(iterations):
        end = (m+1) * subset_length
        start = m * subset_length
        if end > train_length:
            end = train_length

        train_features, train_labels = rearrange(train_file, test_file, start, end, [])
        for genre in genres:
            t_features, train_label = getLabels(genre, train_labels, train_features)
            clf[genre].partial_fit(t_features, train_label, classes=np.array([0, 1]))
            t_features = []
            train_label = []
            gc.collect()
        #print(genre, test_accur)

        train_features = []
        train_labels = []
        t_features = []
        train_label = []
        gc.collect()

    with open('discogs_SGD_Classifiers.txt', 'wb') as data_file:
        pickle.dump(clf, data_file)

    with open('discogs_genre.txt', 'wb') as data_file:
        pickle.dump(main_genres)
    #writeToTsv(genre_labels, subgenre_labels, keys)

    #print(f.keys())