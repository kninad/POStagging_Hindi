import pickle
import structperc

dev_egs = structperc.read_tagging_file('./code/baseline/parsed_test_hi.txt')
file_weights = open('weights.dict', 'r')
weights = pickle.load(file_weights)

sent1 = dev_egs[0]
sent2 = dev_egs[1]

print "\nTag prediction accuracy: \n"
structperc.fancy_eval(dev_egs, weights)

print "\n\nShow predictions: \n"
for i in range(2):
    sent = dev_egs[i]
    predlabs = structperc.predict_seq(sent[0], weights)
    goldlabs = sent[1]
    structperc.show_predictions(sent[0], goldlabs, predlabs)

















