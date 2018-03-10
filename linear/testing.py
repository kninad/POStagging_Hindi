import numpy as np
import pickle
import nltk

from gensim.models.wrappers import FastText
from gensim.models import KeyedVectors

ROOT_DIR = '/home/ninad/Desktop/NlpProj/'
CURR_DIR = ROOT_DIR + 'code/mikolov/GSlim/'

text_file = open(CURR_DIR + "parsed_test_hi.txt","r")
lines = text_file.readlines()

#W = np.load(CURR_DIR + 'Weight_matrix_linalg.npy')
W = np.load(CURR_DIR + 'W_orthog.npy')

# Load the hindi embeds
hi_model = FastText.load_fasttext_format(ROOT_DIR + 'data/embedding/wiki.hi/wiki.hi.bin')
print('Hindi embeds loaded\n')

# Load the english embeds
#en_model = FastText.load_fasttext_format('./data/embedding/wiki.simple/wiki.simple.bin')
en_model = KeyedVectors.load_word2vec_format(ROOT_DIR + 'data/embedding/GoogleNews-vectors-negative300-SLIM.bin',binary=True)
print('English embeds loaded\n')

word_list = []
tag_list = []

hw = nltk.word_tokenize(lines[154])[0]
hvec = hvec = hi_model[hw] 
epred = np.dot(hvec,W)

print(en_model.similar_by_vector(epred))






