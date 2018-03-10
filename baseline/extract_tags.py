text_file = open("./data/parsed_test_hi.txt", "r")
lines = text_file.readlines()

postags = set()

for p in lines:
    if(len(p.split()) > 0):
        ptag = p.split()[1]
        postags.add(ptag)
    else:
        continue

len(postags)



