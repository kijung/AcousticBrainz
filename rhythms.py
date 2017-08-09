from __future__ import division
from script import *
import random



#files2 = parseTsv(tsv = 'acousticbrainz-mediaeval2017-discogs-train-test.tsv')

#files3 = parseTsv(tsv = 'acousticbrainz-mediaeval2017-lastfm-train-train.tsv')
#files4 = parseTsv(tsv = 'acousticbrainz-mediaeval2017-lastfm-train-test.tsv')
#print(lowlevel_features())
#lst = lowlevel_features()
def read(files, features, descriptor):
    for f in files.keys():
        path = './Downloads/acousticbrainz-mediaeval-train/' + f[:2] + '/' + f + '.json'
        if not os.path.isfile(path):
            continue
        with open(path) as data_file:
			c = json.loads(data_file.read())[descriptor]
			d = dict()
			if descriptor == 'rhythm':
				c.pop('beats_position')
			if descriptor == 'lowlevel':
				#print(c.keys())
				for feat in features:
					d[feat] = c[feat]
				files[f].inputFeatures(d)
			c = 0
        data_file = 0

    return files

def split(files, feature, filepath, features = []):
	f = read(files, features, descriptor = feature)
	f_data = dict()
	for t in f.keys():
		f_data[t] = dict()
		f_data[t]['genres'] = f[t].genres
		f_data[t]['features'] = f[t].features
		f[t].genres = []
		f[t].features = []
	writeToFile(f_data, path = filepath)

def split_lowlevel(files, feature, filepath, part):
	keys = list(files.keys())
	features = []
	path = './Downloads/acousticbrainz-mediaeval-train/' + keys[0][:2] + '/' + keys[0] + '.json'
	if not os.path.isfile(path):
		a = 1
	else:
		with open(path) as data_file:
			features = list(json.loads(data_file.read())[feature].keys())
			length = len(features)
			features = features[(part-1) * length//3:part * length//3]
	for f in files.keys():
		path = './Downloads/acousticbrainz-mediaeval-train/' + f[:2] + '/' + f + '.json'
		if not os.path.isfile(path):
			continue
		with open(path) as data_file:
			c = json.loads(data_file.read())[feature]
			d = dict()
			for selected_feature in features:
				d[selected_feature] = c[selected_feature]
			files[f].inputFeatures(d)
			c = 0
		data_file = 0
	f_data = dict()
	for t in f.keys():
		f_data[t] = dict()
		f_data[t]['genres'] = f[t].genres
		f_data[t]['features'] = f[t].features
		f[t].genres = []
		f[t].features = []
	writeToFile(f_data, path = filepath + 'lowlevel' + str(part) + '.json')
#lst = ['beats_count', 'bpm', 'bpm_histogram_first_peak_bpm', 'bpm_histogram_first_peak_spread', 'bpm_histogram_first_peak_weight', 'bpm_histogram_second_peak_bpm', 'bpm_histogram_second_peak_spread', 'bpm_histogram_second_peak_weight', 'beats_loudness', 'beats_loudness_band_ratio', 'onset_rate', 'danceability']
#accur = dict()
files = parseTsv(tsv = 'acousticbrainz-mediaeval2017-discogs-train-train.tsv')
features = ['mfcc', 'melbands', 'gfcc', 'spectral_contrast_coeffs', 'spectral_contrast_valleys']
split(files, 'lowlevel', 'discogs_train_train_lowlevel.json', features=features)

files = parseTsv(tsv = 'acousticbrainz-mediaeval2017-discogs-train-test.tsv')
split(files, 'lowlevel', 'discogs_train_test_lowlevel.json')
#split(files, 'rhythm', 'discogs_train_train_rhythm.json')
#split(files, 'tonal', 'discogs_train_train_tonal.json')
#split_lowlevel(files, 'lowlevel', 'discogs_train_train_', 1)
#split_lowlevel(files, 'lowlevel', 'discogs_train_train_', 2)
#split_lowlevel(files, 'lowlevel', 'discogs_train_train_', 3)
#files = 0
#files = parseTsv(tsv = 'acousticbrainz-mediaeval2017-discogs-train-test.tsv')
#split(files, 'rhythm', 'discogs_train_test_rhythm.json')
#split(files, 'tonal', 'discogs_train_test_tonal.json')
#files = 0
	#split(files2, feature, 'discogs_train_test_')
	#split(files3, feature, 'lastfm_train_train_')
	#split(files4, feature, 'lastfm_train_test_')


#writeToFile(accur, path = 'results_discogs_rfc.json')



		


#stat = stats(f)
#pickleData(files, name = 'discogs_train_test_mean_mfcc.txt')
"""
train = unpickleData(path = "discogs_train_train_mean_mfcc.txt")
test = unpickleData(path = "discogs_train_test_mean_mfcc.txt")
stat = stats(files)
train_stats = stats(train)
test_stats = stats(test)
genre_keys = list(train_stats.keys())
random.shuffle(genre_keys)
for genre in genre_keys[:1]:
	train_features, train_labels = reformat(train, genre)
	test_features, test_labels = reformat(test, genre)
	print(genre)
	classify(train_features, train_labels, test_features, test_labels, genre)
"""
#print(stat)

#print(f.keys())
