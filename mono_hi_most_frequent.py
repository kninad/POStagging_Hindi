import pickle
import numpy as np
from collections import defaultdict

#with open("./data/dict/iitb_word_dict.pkl", "rb") as wfile:
with open("./data/dict/csv_word_dict.pkl", "rb") as wfile:
    hien_dict = pickle.load(wfile)


# MONOLINGUAL CORPORA
# Just a simple monolingual part of a parallel hi-en corpora (iitb cflit)
text_file = open("data//iitb_cfilt/iitb_parallel/parallel/IITB.en-hi.hi", "r")
lines = text_file.readlines()

# Only start reading from this line since lines before are from 
# technical documentation like GNOME and KDE which we don't want.
# basically they are hindi versions of technical english words.
N_begin = 440000 
lines = lines[N_begin:]

#hiWordSet = set(hi_words)
hiWordSet = hien_dict.keys()
hi_countDict = defaultdict(float)

for sent in lines:
    tempList = sent.split()
    for token in tempList:
        if(token in hiWordSet):
            hi_countDict[token] += 1.0

# sorted in D.O
hiDict_sorted = sorted(hi_countDict, key = hi_countDict.get, reverse=True)

#outfile = './data/dict/iitbDict_topFreq_keys.pkl'
outfile = './data/dict/csvDict_topFreq_keys.pkl'
with open(outfile, "wb") as wfile:
    pickle.dump(hiDict_sorted)


