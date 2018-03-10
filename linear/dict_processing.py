import pickle
import csv

########################################

### IIT-B dict ###

text_file = open("data/dict/UW-Hindi_Dict-20131003.txt", "r")
lines = text_file.readlines()

# LIST implementation
hi_words = []
en_words = []

for i in range(len(lines)):
    t = lines[i]
    hword = t[t.find("[")+1 : t.find("]")]
    eword = t[t.find('\"')+1 : t.find("(")]
    if(len(hword.split()) == 1 and len(eword.split()) == 1):
        hi_words.append(hword)
        en_words.append(eword)


## Saving
tup_list = (hi_words, en_words)
with open("./data/dict/iitb_word_lists.pkl", "wb") as wfile:
    pickle.dump(tup_list, wfile)

## Loading
#~ with open("./data/dict/iitb_word_lists.pkl", "rb") as wfile:
    #~ word_tup = pickle.load(wfile)
#~ (hi_words, en_words) = word_tup



## DICT implementation

#~ hi_en = {}
#~ for i in range(len(lines)):
    #~ t = lines[i]
    #~ hword = t[t.find("[")+1 : t.find("]")]
    #~ eword = t[t.find('\"')+1 : t.find("(")]
    #~ if(len(hword.split()) == 1 and len(eword.split()) == 1):
        #~ hi_en[hword] = eword


########################################

### CSV:- HI-EN word lists ###

# LISTS

hi_words = []
en_words = []

with open('./data/dict/English-Hindi Dictionary.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        w_en = row[0]
        w_hi = row[1]
        if(len(w_en.split()) == 1 and len(w_hi.split()) == 1):
            en_words.append(w_en)
            hi_words.append(w_hi)

hi_words = hi_words[1:]
en_words = en_words[1:]

tup_list = (hi_words, en_words)
with open("./data/dict/csv_word_lists.pkl", "wb") as wfile:
    pickle.dump(tup_list, wfile)

## Loading
#~ with open("./data/dict/csv_word_lists.pkl", "rb") as wfile:
    #~ word_tup = pickle.load(wfile)
#~ (hi_words, en_words) = word_tup



## DICT METHOD
hi_en = {} # a dict
with open('./data/dict/English-Hindi Dictionary.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        w_en = row[0]
        w_hi = row[1]
        if(len(w_en.split()) == 1 and len(w_hi.split()) == 1):
            hi_en[w_hi] = w_en  # we only need singletons        


#####







