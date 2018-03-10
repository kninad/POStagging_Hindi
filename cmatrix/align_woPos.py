from __future__ import division
import numpy as np
import pickle
import re
import codecs
#import editdistance
from gensim.models import KeyedVectors
from collections import defaultdict
import string
import nltk

ROOT_DIR = '/home/ninad/Desktop/NlpProj/'
CURR_DIR = ROOT_DIR + 'code/govt_tag_cmatrix/'
DATA_DIR = CURR_DIR + 'data/'

def get_embedding(count_matrix, target_word, model):
    total = count_matrix[target_word]['__TOTAL__']
    model_dimension = model.vector_size
    source_embedding = np.zeros((model_dimension,))
    if target_word in count_matrix:
        source_words_dict = count_matrix[target_word]
        for source_word in source_words_dict:
            if source_word not in model: continue
            count = source_words_dict[source_word]
            source_embedding += count*model[source_word] 

    source_embedding = source_embedding/total
    return model.wv.similar_by_vector(source_embedding, topn=1)[0][0]


def generate_count_matrix():
    
    hindi_english_corpus = ROOT_DIR + "data/HindEnCorp_parallel/HindEnCorp 0.5/hindencorp05.plaintext"
    word_align = ROOT_DIR + "code/gh_codes/Cross-Lingual-POS-Tagging/forward.align"
    f = codecs.open(hindi_english_corpus, 'r', 'utf-8').read().split("\n")
    word_align = codecs.open(word_align, 'r', 'utf-8').read().split("\n")


    count_matrix = defaultdict(lambda: defaultdict(int))
    for index, sentence in enumerate(f):
        sentence = sentence.split('\t')
        if len(sentence)!=5: continue
        english_words = sentence[3].split()
        hindi_words = sentence[4].split()
        alignments = word_align[index].split()
        for idx in alignments:
            target_idx, source_idx = idx.split('-')
            source_idx = int(source_idx)
            target_idx = int(target_idx)
            count_matrix[hindi_words[source_idx]][english_words[target_idx]]+=1
            count_matrix[hindi_words[source_idx]]['__TOTAL__']+=1
    return count_matrix
        

model1 = KeyedVectors.load_word2vec_format(ROOT_DIR + 'data/embedding/GoogleNews-vectors-negative300-SLIM.bin',
                                           binary=True)
ctMatrix = generate_count_matrix()

#~ txtname = 'parsed_train_hi.txt'
txtname = 'clean_hin_entertainment_set1.txt.pkl'
fname = DATA_DIR + txtname

#~ text_file = open(fname, 'r')
#~ lines = text_file.readlines()

with open(fname, 'rb') as rfile:
    lines = pickle.load(rfile)

enWordList = []
it = 1
for l in lines:
    #~ print(it)    
    if(l != '\n'):        
        hw = l.split()[0]
        ew = get_embedding(ctMatrix, hw, model1)
        enWordList.append(ew)
    else:
        enWordList.append('\n')
    #~ it+=1

EnWordFile = DATA_DIR + 'EnWords_' + txtname
with open(EnWordFile, 'wb') as wfile:
    pickle.dump(enWordList, wfile)

nidxs = [] #new line indices
for i in range(len(enWordList)):
    if(enWordList[i]=='\n'):
        nidxs.append(i)        


i1 = 0
tagged_list = []

for j in range(len(nidxs)):
    i2 = nidxs[j]
    esent = enWordList[i1:i2]
    tags = nltk.pos_tag(esent, tagset='universal')
    for j in range(len(tags)):
        tags[j] = list(tags[j])
    if(len(tags)!=0):
        tags[-1][-1] = '.'
    tagged_list.append(tags)
    tagged_list.append('\n')
    i1=i2+1

nlist = [] # making it into similar format as hiTagList
for i in range(len(tagged_list)):
    if(tagged_list[i] != '\n'):
        for wl in tagged_list[i]:
            nlist.append(wl)
    else:
        nlist.append('\n')

EnTagFile = DATA_DIR + 'EnTags_' + txtname
with open(EnTagFile, 'wb') as wfile:
    pickle.dump(nlist, wfile)

#~ hlines = lines
#~ for i in range(len(hlines)):                                                        
    #~ if(hlines[i]!='\n'):                                                                                                  
        #~ hlines[i] = hlines[i].split()

#~ HiTagFile = DATA_DIR + 'HiTags_' + txtname
#~ with open(HiTagFile, 'wb') as wfile:
    #~ pickle.dump(hlines, wfile)      

#~ # Saved the processed lists for reporting statistical measures like accuracy,
#~ # precision-recall, f1score and support

#~ hitags = hlines
#~ entags = nlist



















