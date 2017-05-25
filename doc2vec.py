import gensim
import os
import collections
import smart_open
import random

NEW_DATA = "/home/paruru/graduate/new_data/"

def file_output(filename, listname) :
    fileout = open(NEW_DATA+filename,'w')
    for line in listname :
        for num in line :
            fileout.write(str(num) + " ")
        fileout.write("\n")
    fileout.close()

def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

train_corpus = list(read_corpus(NEW_DATA+"word_list_for_w2v"))

model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=55)

model.build_vocab(train_corpus)

model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)

labeledfile = open(NEW_DATA + "all_lebel_and_content_as_words",'r')
targetfile = open(NEW_DATA + "all_target_content_as_words",'r')

word2numDict = dict()
lebelContentNumLists = list()
index = int(1)
for line in labeledfile :
    li = line.split()
    if li[len(li)-1] == '\n':
        del li[len(li)-1]
    newli = list()
    if word2numDict.has_key(li[0]) :
        newli.append(word2numDict[li[0]])
    else :
        word2numDict[li[0]] = index
        newli.append(index)
        index = index +1
    valueArray = model.infer_vector(li[1:])
    newli.extend(valueArray.tolist())

    lebelContentNumLists.append(newli)

file_output("all_lebel_content_as_digit_d2v_super",lebelContentNumLists)

targetContentNumlist = list()

for line in targetfile :
    li = line.split()
    if li[len(li)-1] == '\n':
        del li[len(li)-1]
    arr = model.infer_vector(li)
    targetContentNumlist.append(arr.tolist())

file_output("all_target_content_as_digit_d2v",targetContentNumlist)

file_output("all_super_label_dict",word2numDict)

targetfile.close()
labeledfile.close()