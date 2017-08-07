from __future__ import division
from script import *
import random
import gc
from sklearn.preprocessing import normalize, StandardScaler


#	files = parseTsv(tsv = 'acousticbrainz-mediaeval2017-discogs-train-train.tsv')
#files2 = parseTsv(tsv = 'acousticbrainz-mediaeval2017-discogs-train-test.tsv')


def lowlevel_features(path = './Downloads/acousticbrainz-mediaeval-train/08/0812194a-2575-4af5-812a-c00054137c7d.json'):
	with open(path) as data_file:
		data = json.loads(data_file.read())
	return list(data['lowlevel'].keys())

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
    for f in files.keys():
        gen = files[f]['genres']
        for g in gen:
            #if '---' in g:
            #    continue
            if g not in stat:
                stat[g] = 1
            else:
                stat[g] += 1
    genres = stat.keys()

    return stat

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

def flatten(l): return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]

def writeToTsv(genre_labels, subgenre_labels, keys):
	combine = []
	for n, key in enumerate(keys):
		detail = []
		detail.append(key)
		for genre in genre_labels:
			if genre_labels[genre][n][0] < genre_labels[genre][n][1]:
				detail.append(genre)
				for subgenre in subgenre_labels[genre]:
					if subgenre_labels[genre][subgenre][n][0] < subgenre_labels[genre][subgenre][n][1]:
						detail.append(subgenre)
		combine.append(detail)
	with open('discogs_train_test_tonal.tsv', 'w') as f:
		for lst in combine:
			f.writelines(('\t'.join(lst) + '\n').encode('utf-8'))

def extractFeatures(features, scalar, mode):
	train_labels = dict()
	train_features = dict()
	#scalar = dict()
	for category in features:
		data = readjson('discogs_train_' + mode + '_' + category + '.json')

		keys = data.keys()
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
	for m in train_features.values():
	#train.append(pad_or_truncate(flatten(m), 57))
		train.append(flatten(m))
	train_features = train	
	return train_features, train_labels.values(), scalar

if __name__ == "__main__":
	#data = readjson('discogs_train_train_rhythm.json')
	data = readjson('discogs_train_train_tonal.json')
	genres = getGenres(data).keys()
	genres.sort()
	#subgenres = []
	main_genres = dict()
	for gen in genres:
		if '---' in gen:
			main_genres[gen.split('---')[0]].append(gen)
		else:
			main_genres[gen] = []

	data = 0
	#gc.collect()
	data = readjson('discogs_train_test_tonal.json')
	keys = list(data.keys())
	data = 0
	#gc.collect()
	#features = data[list(data.keys())[0]]['features'].keys()
	#features = ['beats_count', 'bpm', 'danceability', 'beats_loudness_band_ratio'] #57
	#features = ['bpm_histogram_first_peak_bpm', 'bpm_histogram_first_peak_spread', 'bpm_histogram_first_peak_weight', 'bpm_histogram_second_peak_bpm', 'bpm_histogram_second_peak_spread', 'bpm_histogram_second_peak_weight', 'beats_count', 'bpm', 'beats_loudness', 'danceability', 'beats_loudness_band_ratio']
	features = dict()
	features['tonal'] = ['tuning_frequency', 'thpcp', 'hpcp', 'key_strength', 'chords_strength', 'chords_histogram', 'chords_changes_rate', 'chords_number_rate', 'tuning_diatonic_strength', 'tuning_equal_tempered_deviation', 'tuning_nontempered_energy_ratio']
	features['rhythm'] = ['bpm_histogram_first_peak_bpm', 'bpm_histogram_first_peak_spread', 'bpm_histogram_first_peak_weight', 'bpm_histogram_second_peak_bpm', 'bpm_histogram_second_peak_spread', 'bpm_histogram_second_peak_weight', 'beats_count', 'bpm', 'beats_loudness', 'danceability']
	scalar = dict()
	mode = 'train'
	train_features, train_labels, scalar = extractFeatures(features, scalar, mode)

	#gc.collect()
	mode = 'test'
	test_features, test_labels, scalar = extractFeatures(features, scalar, mode)
	#gc.collect()
	genre_labels = dict()
	subgenre_labels = dict()
	for genre in main_genres.keys():
		t_features, train_label, test_label = getLabels(genre, train_labels, test_labels, train_features)
		valid_accur, test_accur, test_prediction = classify(t_features, train_label, test_features, test_label, genre = genre, classifier = 'RFC')
		print(genre, test_accur)
		#new_prediction = [0 for n in range(len(test_label))]

		# make a copy of test_prediction
		"""
		for pred in test_prediction:
			if pred[0] > pred[1]:
				new_prediction.append(0)
			else:
				new_prediction.append(1)
		"""
		t_features = []
		train_label = []
		test_label = []
		#gc.collect()
		genre_labels[genre] = test_prediction
		subgenre_labels[genre] = dict()
		for subgenre in main_genres[genre]:
			subgenre_labels[genre][subgenre] = dict()
			t_features, train_label, test_label = getLabels(subgenre, train_labels, test_labels, train_features)

			valid_accur, test_accur, subgenre_prediction = classify(t_features, train_label, test_features, test_label, genre = subgenre, classifier = 'SVM')
			#print(subgenre, test_accur)
			subgenre_labels[genre][subgenre] = subgenre_prediction
			t_features = []
			train_label = []
			test_label = []
			#gc.collect()
	train_features = []
	train_labels = []
	test_labels = []
	test_features = []
	gc.collect()

	writeToTsv(genre_labels, subgenre_labels, keys)
	#writeToFile(record, path = 'discogs_results_subgenres_combination.json')
	#stat = stats(f)
	#pickleData(files, name = 'discogs_train_test_mean_mfcc.txt')
	"""
			if record[genre] < test_accur:
				for n, prediction in enumerate(test_prediction):
					#new_prediction[n] = prediction
					if prediction[0] > prediction[1] and subgenre_prediction[n][0] < subgenre_prediction[n][1]  and prediction[0] < subgenre_prediction[n][1]:
						new_prediction[n] = 1
					elif prediction[0] <= prediction[1] and subgenre_prediction[n][0] < subgenre_prediction[n][1]:
						new_prediction[n] = 1
	"""
	"""
		TP, FP, TN, FN = 0, 0, 0, 0
		#print(genre_label[0])
		for n, label in enumerate(genre_label):
			#print(label, new_prediction[n])
			if label == new_prediction[n]:
				if label == 1:
					TP += 1
				else:
					TN += 1
			else:
				if label == 0:
					FP+= 1
				else:
					FN+= 1
		accuracy = float(TP + TN)/ float(TP + TN + FP + FN)
		print('Final Prediction: ', genre, accuracy)
			#record[genre] = test_accur			
	"""
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