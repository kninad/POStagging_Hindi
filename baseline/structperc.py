from __future__ import division
import sys,re,random
from collections import defaultdict
from pprint import pprint
import pickle
##########################
# Stuff you will use

import vit_starter  # your vit.py from part 1

#OUTPUT_VOCAB = set(""" ! # $ & , @ A D E G L M N O P R S T U V X Y Z ^ """.split())

OUTPUT_VOCAB = {'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'NOUN', 'NUM',
                'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'VERB', 'X'}


##########################
# Utilities

def dict_subtract(vec1, vec2):
    """treat vec1 and vec2 as dict representations of sparse vectors"""
    out = defaultdict(float)
    out.update(vec1)
    for k in vec2: out[k] -= vec2[k]
    return dict(out)

def dict_argmax(dct):
    """Return the key whose value is largest. In other words: argmax_k dct[k]"""
    return max(dct.iterkeys(), key=lambda k: dct[k])

def dict_dotprod(d1, d2):
    """Return the dot product (aka inner product) of two vectors, where each is
    represented as a dictionary of {index: weight} pairs, where indexes are any
    keys, potentially strings.  If a key does not exist in a dictionary, its
    value is assumed to be zero."""
    smaller = d1 if len(d1)<len(d2) else d2  # BUGFIXED 20151012
    total = 0
    for key in smaller.iterkeys():
        total += d1.get(key,0) * d2.get(key,0)
    return total

def read_tagging_file(filename):
    """Returns list of sentences from a two-column formatted file.
    Each returned sentence is the pair (tokens, tags) where each of those is a
    list of strings.
    """
    sentences = open(filename).read().strip().split("\n\n")
    ret = []
    for sent in sentences:
        lines = sent.split("\n")
        pairs = [L.split("\t") for L in lines]
        tokens = [tok for tok,tag in pairs]
        tags = [tag for tok,tag in pairs]
        ret.append( (tokens,tags) )
    return ret
###############################

## Evaluation utilties you don't have to change

def do_evaluation(examples, weights):
    num_correct,num_total=0,0
    for tokens,goldlabels in examples:
        N = len(tokens); assert N==len(goldlabels)
        predlabels = predict_seq(tokens, weights)
        num_correct += sum(predlabels[t]==goldlabels[t] for t in range(N))
        num_total += N
    print "%d/%d = %.4f accuracy" % (num_correct, num_total, num_correct/num_total)
    return num_correct/num_total

def fancy_eval(examples, weights):
    confusion = defaultdict(float)
    bygold = defaultdict(lambda:{'total':0,'correct':0})
    for tokens,goldlabels in examples:
        predlabels = predict_seq(tokens, weights)
        for pred,gold in zip(predlabels, goldlabels):
            confusion[gold,pred] += 1
            bygold[gold]['correct'] += int(pred==gold)
            bygold[gold]['total'] += 1
    goldaccs = {g: bygold[g]['correct']/bygold[g]['total'] for g in bygold}
    for gold in sorted(goldaccs, key=lambda g: -goldaccs[g]):
        print "gold %s acc %.4f (%d/%d)" % (gold,
                goldaccs[gold],
                bygold[gold]['correct'],bygold[gold]['total'],)

def show_predictions(tokens, goldlabels, predlabels):
    print "\n%-20s %-4s %-4s" % ("word", "gold", "pred")
    print "%-20s %-4s %-4s" % ("----", "----", "----")
    for w, goldy, predy in zip(tokens, goldlabels, predlabels):
        out = "%-20s %-4s %-4s" % (w,goldy,predy)
        if goldy!=predy:
            out += "  *** Error"
        print out
        

###############################

## YOUR CODE BELOW


def train(examples, stepsize=1, numpasses=10, do_averaging=False, devdata=None):
    """
    IMPLEMENT ME !
    Train a perceptron. This is similar to the classifier perceptron training code
    but for the structured perceptron. Examples are now pairs of token and label
    sequences. The rest of the function arguments are the same as the arguments to
    the training algorithm for classifier perceptron.
    """

    weights = defaultdict(float)
    
    Sdict = defaultdict(float) # Just accumulate the S here.
    t=1

    def get_averaged_weights():
        # IMPLEMENT ME!
        avg_weights = defaultdict(float)
        #T = len(examples) * numpasses
        for k in Sdict.keys():
            #Sdict[k] *= (1/T) # Or should it be (1/t)?
            avg_weights[k] = weights[k] - (1/t) * Sdict[k]
        return avg_weights

    print "[training...]"

    for pass_iteration in range(numpasses):
        print "\n\nTraining iteration %d\n" % pass_iteration
        # IMPLEMENT THE INNER LOOP!
        # Like the classifier perceptron, you may have to implement code
        # outside of this loop as well!
        random.shuffle(examples)
        
        for tokens,goldlabels in examples:
            predlabels = predict_seq(tokens, weights)
            if(predlabels != goldlabels):
                f_true = features_for_seq(tokens, goldlabels)
                f_pred = features_for_seq(tokens, predlabels)
                g = dict_subtract(f_true, f_pred)
                for k in g.keys():
                    weights[k] += stepsize * g[k]
                    Sdict[k] += (t-1) * stepsize * g[k]
        t+=1 # outside the if statement. (still in the outer loop though!)
        


        # Evaluation at the end of a training iter
        print "TR  RAW EVAL:\n",
        do_evaluation(examples, weights)
        if devdata:
            print "DEV RAW EVAL:\n",
            do_evaluation(devdata, weights)
        if devdata and do_averaging:
            print "DEV AVG EVAL:\n",
            do_evaluation(devdata, get_averaged_weights())

    print "Learned weights for %d features from %d examples" % (len(weights), len(examples))

    # NOTE different return value then classperc.py version.
    return weights if not do_averaging else get_averaged_weights()

def predict_seq(tokens, weights):
    """
    IMPLEMENT ME!
    takes tokens and weights, calls viterbi and returns the most likely
    sequence of tags
    """
    N = len(tokens)
    retval = calc_factor_scores(tokens, weights)
    Ascores = retval[0]
    Bscores = retval[1]    
    # once you have Ascores and Bscores, could decode with
    
    #predlabels = greedy_decode(Ascores, Bscores, OUTPUT_VOCAB)
    predlabels = vit_starter.viterbi(Ascores, Bscores, OUTPUT_VOCAB)
    
    return predlabels

def greedy_decode(Ascores, Bscores, OUTPUT_VOCAB):
    """Left-to-right greedy decoding.  Uses transition feature for prevtag to curtag."""
    N=len(Bscores)
    if N==0: return []
    out = [None]*N
    out[0] = dict_argmax(Bscores[0])
    for t in range(1,N):
        tagscores = {tag: Bscores[t][tag] + Ascores[out[t-1], tag] for tag in OUTPUT_VOCAB}
        besttag = dict_argmax(tagscores)
        out[t] = besttag
    return out

def local_emission_features(t, tag, tokens):
    """
    Feature vector for the B_t(y) function
    t: an integer, index for a particular position
    tag: a hypothesized tag to go at this position
    tokens: the list of strings of all the word tokens in the sentence.
    Retruns a set of features.
    """
    curword = tokens[t]
    feats = defaultdict(float)
    feats["tag=%s_biasterm" % tag] = 1
    feats["tag=%s_curword=%s" % (tag, curword)] = 1
    
    #FEAT-1
    #fchar = curword[0] #For first char       
    #if(fchar == '@' or fchar == '#'):
    #    feats["tag=%s_fchar=%s" % (tag, fchar)] = 1
    #else:
        #feats["tag=%s_fchar=%s" % (tag, fchar)] = 0
    
    #FEAT-2
    #f4chars = tokens[0:4] if len(tokens)>=4 else 'NULL'
    #if(f4chars == 'http'):
        #feats["tag=%s_f4http=%s" % (tag, f4chars)] = 1
    #else:
        #feats["tag=%s_f4http=%s" % (tag, f4chars)] = 0
    
    #FEAT-3
    nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for i in nums:
        if i in tokens == True:
            feats["tag=%s_num=%s" % (tag, i)] += 1
        else:
            feats["tag=%s_num=%s" % (tag, i)] += 0

    return feats

def features_for_seq(tokens, labelseq):
    
    """
    IMPLEMENT ME!

    tokens: a list of tokens
    labelseq: a list of output labels
    The full f(x,y) function. Returns one big feature vector. This is similar
    to features_for_label in the classifier peceptron except here we aren't
    dealing with classification; instead, we are dealing with an entire
    sequence of output tags.

    This returns a feature vector represented as a dictionary.
    """
    feat_vec = defaultdict(float)      
    
    for t in range(len(tokens)):
        tempdict = local_emission_features(t, labelseq[t], tokens)
        #totvals = sum(tempdict.values()) - 1
        #feat_vec[(labelseq[t], tokens[t])] += totvals
        for k in tempdict.keys():
            feat_vec[k] += tempdict[k]

        #feat_vec["tag=%s_curword=%s" % (labelseq[t], tokens[t])] += tempdict["tag=%s_curword=%s" % (labelseq[t], tokens[t])]
        
        if(t != len(tokens)-1):
            feat_vec[(labelseq[t], labelseq[t+1])] += 1
        else:
            continue

    return feat_vec


def calc_factor_scores(tokens, weights):
    """
    IMPLEMENT ME!

    tokens: a list of tokens
    weights: perceptron weights (dict)

    returns a pair of two things:
    Ascores which is a dictionary that maps tag pairs to weights
    Bscores which is a list of dictionaries of tagscores per token
    """
    N = len(tokens)
    # MODIFY THE FOLLOWING LINE
    #Ascores = { (tag1,tag2): 0 for tag1 in OUTPUT_VOCAB for tag2 in OUTPUT_VOCAB }
    Ascores = { (tag1,tag2): weights[(tag1,tag2)]  for tag1 in OUTPUT_VOCAB for tag2 in OUTPUT_VOCAB }
    Bscores = [defaultdict(float) for i in range(N)]
    
    for t in range(N):
        # IMPLEMENT THE INNER LOOP    
        #Bscores[t] = defaultdict(float)
        for tag in OUTPUT_VOCAB:
            tempdict = local_emission_features(t, tag, tokens)
            Bscores[t][tag] = dict_dotprod(tempdict,weights)

    assert len(Bscores) == N
    return Ascores, Bscores

if __name__ == '__main__':
    # You may implement your code here    
    train_egs = read_tagging_file('./code/baseline/parsed_train_hi.txt')
    dev_egs = read_tagging_file('./code/baseline/parsed_test_hi.txt')
    soldict = train(train_egs, numpasses=20, do_averaging=True, devdata=dev_egs)
    file_weights = open('./code/baseline/weights.dict', 'w')
    pickle.dump(soldict, file_weights)            
