
import numpy as np
import pickle

from gensim.models.wrappers import FastText
from gensim.models import KeyedVectors

ROOT_DIR = '/home/ninad/Desktop/NlpProj/'
CURR_DIR = 'code/mikolov/GSlim'

with open(ROOT_DIR + "data/dict/csv_word_dict.pkl", "rb") as wfile:
    hien_dict = pickle.load(wfile)
#(hi_words, en_words) = word_tup

with open(ROOT_DIR + "data/dict/csvDict_topFreq_keys.pkl", "rb") as wfile:
    hiDict_sKeys = pickle.load(wfile)

# Top hindi words 
up_bound = 6000
topKeys = hiDict_sKeys[0:up_bound] 
# Corresponding english words obtained from the dictionary
en_maps = [hien_dict[topKeys[i]] for i in range(len(topKeys))]
print('En maps obtained\n')

# Load the hindi embeds
hi_model = FastText.load_fasttext_format(ROOT_DIR + 'data/embedding/wiki.hi/wiki.hi.bin')
print('Hindi embeds loaded\n')

# Load the english embeds
#en_model = FastText.load_fasttext_format('./data/embedding/wiki.simple/wiki.simple.bin')
en_model = KeyedVectors.load_word2vec_format(ROOT_DIR + 'data/embedding/GoogleNews-vectors-negative300-SLIM.bin',
                                            binary=True)
print('English embeds loaded\n')

N = len(topKeys)
D = 300 # dimension of word vectors in w2vec
hi_vecs = np.zeros([300, N])
en_vecs = np.zeros([300, N])
err_idxs = []
for i in range(N):    
    hw = topKeys[i]  
    #ew = hien_dict[hw]
    ew = en_maps[i]
    try:
        hi_vecs[:,i] = hi_model[hw]
        en_vecs[:,i] = en_model[ew]
        #continue
    except KeyError:
        err_idxs.append(i) 
        # Add the index for either of the hindi or english
        # that give a key error.
        
print('Crude dict mapping obtained\n')

# remove the all-zero entries from the vector data
# basically, words which gave keyError in either of
# dictionary (word-embeddings)
# err_idxs indicate the column indices
err_arr = np.array(err_idxs)
hi_vecs = np.delete(hi_vecs, err_arr, axis = 1)
en_vecs = np.delete(en_vecs, err_arr, axis = 1)
print('Removed vectors at error indices\n')

# remove the word from lists too
final_hi_words = [topKeys[i] for i in range(len(topKeys)) if i not in err_idxs]
final_en_words = [en_maps[i] for i in range(len(en_maps)) if i not in err_idxs]

with open('./new1/final_hiWords.pkl', 'wb') as wfile:
    pickle.dump(final_hi_words, wfile)

with open('./new1/final_enWords.pkl', 'wb') as wfile:
    pickle.dump(final_en_words, wfile)

hiVec_outfile = ROOT_DIR + "code/mikolov/GSlim/new_hivecs_csv.npy"
np.save(hiVec_outfile, hi_vecs)

enVec_outfile = ROOT_DIR + "code/mikolov/GSlim/new_envecs_csv.npy"
np.save(enVec_outfile, en_vecs)    
print('Hindi and English vectors saved\n')




##### code dump ######

#~ hi_model = KeyedVectors.load_word2vec_format('./data/embedding/wiki.hi/wiki.hi.vec')
#~ en_model = KeyedVectors.load_word2vec_format('./data/embedding/wiki.simple/wiki.simple.vec')


# Getting the tokens 
#~ fast_hi_words = []
#~ for word in hi_model.vocab:
    #~ fast_hi_words.append(word)

#~ fast_en_words = []
#~ for word in en_model.vocab:
    #~ fast_en_words.append(word)

#~ counts = 0
#~ for w in hi_words:
    #~ if(w in hi_model.vocab):
        #~ counts+=1


#~ for i in range(len(hi_words)):
    #~ hw = hi_words[i]    
    #~ if(hw in hi_model.vocab):
        #~ hi_vecs[:,i] = hi_model[hw]
    #~ else:
        #~ tList = hi_model.similar_by_word(hw, topn=1)
        #~ tWord = (tList[0])[0]  # most similar word
        #~ hi_vecs[:,i] = hi_model[tWord]
