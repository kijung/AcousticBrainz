#!/bin/bash
source ~/environment/bin/activate
#python store.py -i discogs
#python store.py -i lastfm
python store.py -i tagtraum
python labels.py -i tagtraum
python scale.py -i tagtraum
