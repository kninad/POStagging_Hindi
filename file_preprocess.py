'''
Script to clean the hindi pos tagged files and arrange them in pre-defined 
format. Note that two files (argiculture and entertainment) DO NOT have POS tag
information with them, but I have still processed them.
'''


import pickle
import csv

ROOT_DIR = '/home/ninad/Desktop/NlpProj/'
CURR_DIR = ROOT_DIR + 'data/sample_posTags_Govt/Hindi_monolingual_sampledata/'

def clean_govt_files(filepath):
    text_file = open(filepath, "r")
    lines = text_file.readlines()
    lines = lines[1:] # first line is not useful
    parsed_list = [] 
    for sent in lines:
        # first token in sent is ID value, so will ignore it
        split_sent = sent.split()
        split_sent = split_sent[1:]
        for word in split_sent:
            tag_idx = word.find('\\') #POS-tag is specified by a '\' char before it
            hi_word = word[ :tag_idx]
            hi_tag = word[tag_idx+1: ]
            parsed_list.append(hi_word + '\t' + hi_tag)
        
        parsed_list.append('\n') #Insert newline after each sentence
    
    return parsed_list

def clean_govt_woPOS(filepath):
    text_file = open(filepath, "r")
    lines = text_file.readlines()
    lines = lines[1:] # first line is not useful
    parsed_list = [] 
    for sent in lines:
        # first token in sent is ID value, so will ignore it
        split_sent = sent.split()
        split_sent = split_sent[1:]
        for word in split_sent:
            #~ tag_idx = word.find('\\') #POS-tag is specified by a '\' char before it
            #~ hi_word = word[ :tag_idx]
            #~ hi_tag = word[tag_idx+1: ]
            parsed_list.append(word)
        
        parsed_list.append('\n') #Insert newline after each sentence
            
    return parsed_list


fname = "hin_entertainment_set1.txt"
fpath = CURR_DIR + fname
#cleaned_list = clean_govt_files(fpath)
cleaned_list = clean_govt_woPOS(fpath)
out_fname = "clean_" + fname + ".pkl"

with open(CURR_DIR + out_fname, 'wb') as outFile:
    pickle.dump(cleaned_list, outFile)
