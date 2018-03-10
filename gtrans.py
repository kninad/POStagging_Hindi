from googletrans import Translator

translator = Translator()
#translator.translate('word',dest='hi').text


with open('./parsed_test_hi.txt', 'r') as tfile1:
    lines = tfile1.readlines()

hi_words = []

for i in range(len(lines)):
    w = lines[i]
    idx = w.find('\t')
    hi_words.append(w[:idx])


for i in range(len(hi_words)):
    print(i)
    tmp = translator.translate(hi_words[i], dest='en').tex
    en_words.append(tmp)



outfile = open(ofname, 'w')
for w in hi_words:
    outfile.write('%s\n' %w)


