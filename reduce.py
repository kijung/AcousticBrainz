from __future__ import division
from script import *
import random
import gc
import pickle
from sklearn.preprocessing import normalize, StandardScaler


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

def extractFeatures(features, scalar, mode, keys):
    train_labels = dict()
    train_features = dict()
    #scalar = dict()
    for category in features:
        data = readjson('discogs_train_' + mode + '_' + category + '.json')

        for key in keys:
            if key not in train_features:
                train_features[key] = []
                train_labels[key] = data[key]['genres']
                data[key]['genres'] = []

        for feature in features[category]:
            mat = []
            sample = data[keys[0]]['features'][feature]
            length = 0
            if not isinstance(sample, float) and not isinstance(sample, int):
                if isinstance(sample, dict):
                    length = len(flatten(sample.values()))
                else:
                    length = len(sample)
            for key in keys:
                c = data[key]['features'][feature]
                if isinstance(c, float) or isinstance(c, int):
                    #train_features[key].append(c)
                    mat.append([c])
                elif isinstance(c, dict):
                    d = flatten(c.values())
                    if len(d) < length:
                        mean = np.mean(np.array(d))
                        d += [mean for n in range(length - len(d))]
                    mat.append(d)   
                else:
                    #train_features[key].append(normalize(np.array(c).reshape(1, -1)[0].tolist()))
                    #train_features[key].append(flatten(c.values()))
                    if len(c) < length:
                        mean = np.mean(np.array(c))
                        c += [mean for n in range(length - len(c))]
                    mat.append(c)
            print(feature, np.shape(mat))
            if feature not in scalar:
                scalar[feature] = StandardScaler().fit(mat)
            mat = scalar[feature].fit_transform(mat)
            for n, key in enumerate(data.keys()):
                train_features[key].append(list(mat[n]))
                mat[n] = 0
                #gc.collect()
            mat = 0
            #gc.collect()

        data = 0
    train = []
    for d in keys:
        m = train_features[d]
    #train.append(pad_or_truncate(flatten(m), 57))
        train.append(flatten(m))
    train_features = train  
    return train_features, train_labels.values(), scalar

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

def rearrange(train_tsv, test_tsv, train_subset_size, test_subset_size):
    files = processTsv(train_tsv)
    lst = list(files.keys())
    random.shuffle(lst)
    lst = lst[:train_subset_size]
    #files = 0
    train = dict()
    train_labels = []

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
            """
            for descriptor in train[f].keys():
                for d in train[f][descriptor].keys():
                    feature = flatten(train[f][descriptor][d])
                    if isinstance(feature, dict):
                        feature = feature.values()
                        length[d] = max(length[d], len(flatten(feature)))
                    elif isinstance(feature, list):
                        length[d] = max(length[d], len(flatten(feature)))
                    elif isinstance(feature, float) or isinstance(feature, int):
                         length[d] = max(length[d], [feature])
            """
    files = 0
    with open('discogs_scalar.txt', 'rb') as data_file:
        scalar = pickle.load(data_file)
    features = list(train[lst[0]]['tonal'].keys()) + list(train[lst[0]]['rhythm'].keys()) + list(train[lst[0]]['lowlevel'].keys())
    train_features = [[] for n in range(len(lst))]
    indicies = ['' for n in range(3064)]
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
        mat = scalar[feature].transform(mat)
        #print(np.shape(mat))
        for n, key in enumerate(mat):
            train_features[n] += list(mat[n])
    #--------------test_features time-----------------#
    print('finished with train')
    print(np.shape(train_features))
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
        length = len(scalar[feature].mean_)
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
            if len(c) < length:
                c += [np.mean(np.array(c)) for n in range(length - len(c))]
            mat.append(c)
        #scalar[feature] = StandardScaler().fit(mat)
        mat = scalar[feature].transform(mat)
        for n, key in enumerate(mat):
            test_features[n] += list(mat[n])
    print(np.shape(test_features))
    return train_features, train_labels, test_features, test_labels, lst, indicies


if __name__ == "__main__":
    specific = 'discogs'
    train_file = 'acousticbrainz-mediaeval2017-' + specific + '-train-train.tsv'
    test_file = 'acousticbrainz-mediaeval2017-' + specific + '-train-test.tsv'

    subset_size = 80000
    #train_features, train_labels, test_features, test_labels, test_keys, indicies = rearrange(train_file, test_file, subset_size, 2000)
    #test_keys, test_features, test_labels = rearrange(test_file, 1000)


    genres = ['rock', 'electronic', 'pop', 'jazz']

    #weights = [0.0 for n in range(2647)]
    weights = dict()
    for genre in genres:
        weights[genre] = np.zeros(2647)

    iterations = 10
    for n in range(iterations):
        train_features, train_labels, test_features, test_labels, test_keys, _ = rearrange(train_file, test_file, subset_size, 2000)
        for genre in genres:
        
            t_features, train_label, test_label = getLabels(genre, train_labels, test_labels, train_features)
            valid_accur, test_accur, test_prediction, importance = classify(t_features, train_label, test_features, test_label, genre = genre, classifier = 'RFC')
            #print(test_accur)
            #print(importance)
            weights[genre] += (np.array(importance))
            t_features, train_label, test_label = 0, 0, 0
            test_prediction = 0
            gc.collect()
        train_features, train_labels, test_features, test_labels, test_keys = 0, 0, 0, 0, 0
        gc.collect()
    

    for genre in genres:
        w = weights[genre]
        w /= iterations
        a = zip(list(w), [n for n in range(2647)])
        a.sort(key = lambda tup:tup[0], reverse=True)
        weights[genre] = a

    with open('weights.txt', 'w') as data_file:
        pickle.dump(weights, data_file)

    """
    a = zip(weights, [n for n in range(2647)])
    a.sort(key = lambda tup:tup[0])
    print(a)
    b = zip(*a)[1]
    print(indicies[b[0]], b[0])
    """
