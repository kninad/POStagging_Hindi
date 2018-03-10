## POS tagging techniques for Hindi

### Plan
Initially, I will focus on implementing a linear transformation from Hindi word vectors to English word vectors. Then for a given Hindi word, its corresponding word vector embedding can be found in the English vector space after which we can retrieve the most similar English word and its corresponding POS-tag.

The key assumption is that POS-tags do not change by much across languages (we are also using a relatively short lexicon to construct the linear transformation between the word vector spaces of the two languages). The pre-trained word vector embeddings for both the languages are taken from Facebook's [fastText](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)

### Update: 2017-11-16
Another assumption for going just for a linear transformation is that the word vectors for similar words in two different languages when visualized (probably by applying PCA), seem to show exhibit the existence of a linear transformation. If this is not true for a pair of languages, then it may better to find the transformation by training a neural network which has inherent non-linearities in it. 

This is probably why I am guessing a linear transformation is not able to correctly able to fit to my Hindi-English word vector data.

### Report
The project report is available [here](https://ninception.github.io/docs/NLP585_final.pdf).
