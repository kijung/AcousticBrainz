import csv

def processTsv(tsv = 'acousticbrainz-mediaeval2017-discogs-train-train.tsv'):
    #a = # of entries, b = filter
    files = dict()
    with open(tsv) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        #next(tsvreader, None)
        for line in tsvreader:
            while '' in line:
                line.remove('')
            #audio = Audio(line[0], line[1], line[2:])
            files[line[0]] = line[1:]
    return files


specific = 'tagtraum'
a = processTsv(specific + '_test_SGD.tsv')
new_tsv = []
genres = dict()
for key in a.keys():
    detail = [key]
    c = a[key]
    c.sort()
    d = []
    genres = dict()
    for s in c:
    	if '---' in s and s.split('---')[0] in genres:
    	    d.append(s)
    	elif '---' not in s:
    	    d.append(s)
    	    genres[s] = 0
    detail += d
    new_tsv.append(detail)

with open(specific + '_test_sorted_SGD.tsv', 'w') as f:
    for lst in new_tsv:
        f.writelines(('\t'.join(lst) + '\n').encode('utf-8'))