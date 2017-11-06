from __future__ import division
from script import *
import argparse
import random
import gc
from sklearn.preprocessing import normalize, StandardScaler, MultiLabelBinarizer
import pickle
import constants
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
def writeToFile(data, path = 'results.json'):
    with open(path, 'w') as f:
        json.dump(data, f)
    #print(lowlevel_features())

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

def train(part, classifier):
    with open(constants.path + specific + part + '_' + 'train.pkl', 'rb') as data_file:
        data = pickle.load(data_file)
    classifier.partial_fit(data['features'], data['genres'])
    return classifier
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
    
    path = constants.path + specific + '_mlb.pkl'
    with open(path, 'rb') as data_file:
        mlb = pickle.load(data_file)

    classifier = MultiOutputClassifier(LinearSVC(C=10, class_weight='balanced', dual=True))
    for n in range(5):
        classifier = train(str(n), classifier)

    path = constants.path + specific + '_classifier.pkl'
    with open(path, 'wb') as data_file:
        pickle.dump(classifier, data_file)   