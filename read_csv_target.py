#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import string
from pattern.en import lemma
import csv
from pandas import Series, DataFrame

dataframes = {
    "test": pd.read_csv("data/test.csv"),
}

uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'

print("000000")

def stripTagsAndUris(x):
    if x:
        # BeautifulSoup on content
        soup = BeautifulSoup(x, "html.parser")
        # Stripping all <code> tags with their content if any
        if soup.code:
            soup.code.decompose()
        # Get all the text out of the html
        text =  soup.get_text()
        # Returning text stripping out all uris
        return re.sub(uri_re, "", text)
    else:
        return ""

for df in dataframes.values():
    df["content"] = df["content"].map(stripTagsAndUris)

print("11111")

def removePunctuation(x):
    # Lowercasing all words
    x = x.lower()
    # Removing non ASCII chars
    x = re.sub(r'[^\x00-\x7f]',r' ',x)
    # Removing (replacing with empty spaces actually) all the punctuations
    return re.sub("["+string.punctuation+"]", " ", x)

for df in dataframes.values():
    df["title"] = df["title"].map(removePunctuation)
    df["content"] = df["content"].map(removePunctuation)
    
print("22222")

stops = set(stopwords.words("english"))
def removeStopwords(x):
    # Removing all the stopwords
    filtered_words = [word for word in x.split() if word not in stops]
    return " ".join(filtered_words)

for df in dataframes.values():
    df["title"] = df["title"].map(removeStopwords)
    df["content"] = df["content"].map(removeStopwords)

print("33333")
def lemmawords(y) :
    for i in range(0,len(y)) :
        y[i] = lemma(y[i])
    return y

for df in dataframes.values():
# From a string sequence of tags to a list of tags
    df["title"] = df["title"].map(lambda x: x.split())
    df["title"] = df["title"].map(lemmawords)
    df["content"] = df["content"].map(lambda x: x.split())
    df["content"] = df["content"].map(lemmawords)

newdic = dict()
newlist = list()

file_wt = open("new_data/target_content_as_words",'w')
file_wta = open("new_data/word_list_for_w2v",'a')

for df in dataframes.values() :
    for i in range (0, df['title'].size) :
        t2 = list()
        t2.extend(df['title'][i])
        t2.extend(df['content'][i])
        for t in t2 :
            file_wt.write(t+' ')
            file_wta.write(t+' ')
        file_wt.write("\n")
                    
file_wt.close()

for name, df in dataframes.items():
    # Saving to file
    df.to_csv("new_data/"+name + "_light_target.csv", index=False)