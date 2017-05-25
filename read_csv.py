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
    "cooking": pd.read_csv("data/cooking.csv"),
    "crypto": pd.read_csv("data/crypto.csv"),
    "robotics": pd.read_csv("data/robotics.csv"),
    "biology": pd.read_csv("data/biology.csv"),
    "travel": pd.read_csv("data/travel.csv"),
    "diy": pd.read_csv("data/diy.csv"),
}

uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'

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
    

stops = set(stopwords.words("english"))
def removeStopwords(x):
    # Removing all the stopwords
    filtered_words = [word for word in x.split() if word not in stops]
    return " ".join(filtered_words)

for df in dataframes.values():
    df["title"] = df["title"].map(removeStopwords)
    df["content"] = df["content"].map(removeStopwords)

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
    df["tags"] = df["tags"].map(lambda x: x.split())
    df["tags"] = df["tags"].map(lemmawords)

print(dataframes["cooking"].iloc[1])

newdic = dict()
newlist = list()

file_wt = open("new_data/all_lebel_and_content_as_words",'w')
file_w2v = open("new_data/all_word_list_for_w2v",'w')

for df in dataframes.values() :
    for i in range (0, df['tags'].size) :
        t2 = list()
        t2.extend(df['tags'][i])
        t2.extend(df['title'][i])
        t2.extend(df['content'][i])
        for t in t2 :
                file_w2v.write(t+' ')
        file_w2v.write("\n")
        
        for tag in df['tags'][i] :
            temp = list()
            temp.append(tag)
            temp.extend(df['title'][i])
            temp.extend(df['content'][i])
            newlist.append(temp)
            for t in temp :
                file_wt.write(t+' ')
            file_wt.write("\n")
                
file_wt.close()
file_w2v.close()

for name, df in dataframes.items():
    # Saving to file
    df.to_csv("new_data/"+name + "_light.csv", index=False)