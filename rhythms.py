from script import *
import random



files = parseTsv(tsv = 'acousticbrainz-mediaeval2017-discogs-train-train.tsv')
#files2 = parseTsv(tsv = 'acousticbrainz-mediaeval2017-discogs-train-test.tsv')

#files3 = parseTsv(tsv = 'acousticbrainz-mediaeval2017-lastfm-train-train.tsv')
#files4 = parseTsv(tsv = 'acousticbrainz-mediaeval2017-lastfm-train-test.tsv')
#print(lowlevel_features())
lst = lowlevel_features()
def read(files, descriptor, feature):
    for f in files.keys():
        path = './Downloads/acousticbrainz-mediaeval-train/' + f[:2] + '/' + f + '.json'
        if not os.path.isfile(path):
            continue
        with open(path) as data_file:
			c = json.loads(data_file.read())[descriptor]
			if descriptor == 'rhythm':
				c.pop('beats_position')
			if descriptor == 'lowlevel':
				
			files[f].inputFeatures(c)
			c = 0
        data_file = 0

    return files

def split(files, feature, filepath):
	f = read(files, descriptor = feature, feature = '')
	f_data = dict()
	for t in f.keys():
		f_data[t] = dict()
		f_data[t]['genres'] = f[t].genres
		f_data[t]['features'] = f[t].features
		f[t].genres = []
		f[t].features = []
	writeToFile(f_data, path = filepath)



#lst = ['beats_count', 'bpm', 'bpm_histogram_first_peak_bpm', 'bpm_histogram_first_peak_spread', 'bpm_histogram_first_peak_weight', 'bpm_histogram_second_peak_bpm', 'bpm_histogram_second_peak_spread', 'bpm_histogram_second_peak_weight', 'beats_loudness', 'beats_loudness_band_ratio', 'onset_rate', 'danceability']
#accur = dict()
split(files, 'lowlevel', 'discogs_train_train_lowlevel.json')
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
