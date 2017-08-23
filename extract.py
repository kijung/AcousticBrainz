from __future__ import division
from script import *
import random
import gc
from sklearn.preprocessing import normalize, StandardScaler
import pickle

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


def saveFeatures(specific,train_tsv, start, end, scalar):
    data = readjson('./Downloads/acousticbrainz-mediaeval-train/01/01330cae-7f9e-451c-a482-2f27d5d3da1d.json')
    features = list(data['lowlevel'].keys()) + list(data['tonal'].keys()) + list(data['rhythm'].keys())
    #print(features)
    features.remove('key_key')
    features.remove('key_scale')
    features.remove('chords_key')
    features.remove('chords_scale')
    features.remove('beats_position')
    
    data = 0
    data = processTsv(train_tsv)
    keys = list(data.keys())
    genres = []
    for key in keys:
        genres.append(data[key])
    with open('discogs_ids.txt', 'w') as f:
        pickle.dump(genres, f)
    keys = keys[start:end]
    genres = 0
    data = 0
    length = dict()
    scalar = dict()

    for f in keys:
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

    for feature in features:
        data = []
        mat = []
        length[feature] = 0
        for f in keys:
            sample = train[f]
            c = 0
            if feature in sample['lowlevel'].keys():
                c = sample['lowlevel'][feature]
            elif feature in sample['tonal'].keys():
                c = sample['tonal'][feature]
            else:
                c = sample['rhythm'][feature]

            if isinstance(c, dict):
                c = flatten(c.values())
            elif isinstance(c, float) or isinstance(c, int):
                c = [c]
            length[feature] = max(len(c), length[feature])
            data.append(c)

        for d in data:
            if len(d) < length[feature]:
                d += [np.mean(d) for n in range(length[feature]-len(d))]
            mat.append(d)
        scalar[feature] = StandardScaler().partial_fit(mat)
        data = 0
        mat = 0
        gc.collect()
        
    for f in keys:
        train[f] = 0
    gc.collect()
    """
    for feature in features:
        print(feature)
        length[feature] = 0
        data = []
        mat = []
        for f in keys:
            path = './Downloads/acousticbrainz-mediaeval-train/' + f[:2] + '/' + f + '.json'
            if not os.path.isfile(path):
                continue
            with open(path) as data_file:
                sample = json.loads(data_file.read())
                c = 0
                if feature in sample['lowlevel'].keys():
                    c = sample['lowlevel'][feature]
                elif feature in sample['tonal'].keys():
                    c = sample['tonal'][feature]
                else:
                    c = sample['rhythm'][feature]
            if isinstance(c, dict):
                c = flatten(c.values())
            elif isinstance(c, float) or isinstance(c, int):
                c = [c]
            length[feature] = max(len(c), length[feature])
            data.append(c)
        for d in data:
            if len(d) < length[feature]:
                d += [np.mean(d) for n in range(length[feature]-len(d))]
            mat.append(d)
        scalar[feature] = StandardScaler().partial_fit(mat)
        data = 0
        mat = 0
    """
    return scalar


            #train[f]['rhythm'].pop('beats_loudness')
if __name__ == "__main__":
    specific = 'discogs'
    train_file = 'acousticbrainz-mediaeval2017-' + specific + '-train-train.tsv'
    test_file = 'acousticbrainz-mediaeval2017-' + specific + '-train-test.tsv'

    files = processTsv(train_file)
    train_length = len(files)
    train = dict()
    lst = list(files.keys())
    f = lst[0]
    path = './Downloads/acousticbrainz-mediaeval-train/' + f[:2] + '/' + f + '.json'

    with open(path) as data_file:
        train[f] = json.loads(data_file.read())
        train[f]['tonal'].pop('key_key')
        train[f]['tonal'].pop('key_scale')
        train[f]['tonal'].pop('chords_key')
        train[f]['tonal'].pop('chords_scale')
        train[f]['rhythm'].pop('beats_position')

    features = list(train[f]['lowlevel'].keys()) + list(train[f]['tonal'].keys()) + list(train[f]['rhythm'].keys()) 
    scalar = dict()

    for f in features:
        scalar[f] = StandardScaler()

    for c in range(train_length//10 + 1):
        print(c)
        start = c*(train_length//10)
        end = (c+1)*(train_length//10)
        if end > train_length:
            end = train_length
        if start >= train_length:
            break
        scalar = saveFeatures(specific, train_file, start, end, scalar)
    with open(specific + '_scalar.txt', 'wb') as data_file:
        pickle.dump(scalar)
