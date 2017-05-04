import io
import sys
import os
import numpy as np
import pandas as pd
import nltk
import gensim
import csv, collections
from textblob import TextBlob
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
import pickle
import replace_emoji

class load_senti_word_net(object):
  
    def __init__(self):
        sent_scores = collections.defaultdict(list)
        with io.open("/home/fmazhar/Downloads/SentiWordNet_3.0.0_20130122.txt") as fname:
            file_content = csv.reader(fname, delimiter='\t',quotechar='"')
            
            for line in file_content:                
                if line[0].startswith('#') :
                    continue                    
                pos, ID, PosScore, NegScore, synsetTerms, gloss = line
                for terms in synsetTerms.split(" "):
                    term = terms.split("#")[0]
                    term = term.replace("-","").replace("_","")
                    key = "%s/%s"%(pos,term.split("#")[0])
                    try:
                        sent_scores[key].append((float(PosScore),float(NegScore)))
                    except:
                        sent_scores[key].append((0,0))
                    
        for key, value in sent_scores.iteritems():
            sent_scores[key] = np.mean(value,axis=0)
        
        self.sent_scores = sent_scores
        
    def score_word(self, word):
        pos = nltk.pos_tag([word])[0][1]
        return self.score(word, pos)   
        
    def score(self,word, pos):    
        if pos[0:2] == 'NN':
            pos_type = 'n'
        elif pos[0:2] == 'JJ':
            pos_type = 'a'
        elif pos[0:2] =='VB':
            pos_type='v'
        elif pos[0:2] =='RB':
            pos_type = 'r'
        else:
            pos_type =  0
            
        if pos_type != 0 :    
            loc = pos_type+'/'+word
            score = self.sent_scores[loc]
            if len(score)>1:
                return score
            else:
                return np.array([0.0,0.0])
        else:
            return np.array([0.0,0.0])
        
        def score_sentencce(self, sentence):
        pos = nltk.pos_tag(sentence)
        print (pos)
        mean_score = np.array([0.0, 0.0])
        for i in range(len(pos)):
            mean_score += self.score(pos[i][0], pos[i][1])
            
        return mean_score
    
    def pos_vector(self, sentence):
        pos_tag = nltk.pos_tag(sentence)
        vector = np.zeros(4)
        
        for i in range(0, len(pos_tag)):
            pos = pos_tag[i][1]
            if pos[0:2]=='NN':
                vector[0] += 1
            elif pos[0:2] =='JJ':
                vector[1] += 1
            elif pos[0:2] =='VB':
                vector[2] += 1
            elif pos[0:2] == 'RB':
                vector[3] += 1
                
        return vector
        
    def gram_features(features,sentence):
      sentence_rep = replace_emoji.replace_reg(str(sentence))
      token = nltk.word_tokenize(sentence_rep)
      token = [porter.stem(i.lower()) for i in token]        
    
      bigrams = nltk.bigrams(token)
      bigrams = [tup[0] + ' ' + tup[1] for tup in bigrams]
      grams = token + bigrams
      #print (grams)
      for t in grams:
        features['contains(%s)'%t]=1.0
        
    def sentiment_extract(features, sentence):
    sentence_rep = replace_emoji.replace_reg(sentence)
    token = nltk.word_tokenize(sentence_rep)    
    token = [porter.stem(i.lower()) for i in token]   
    mean_sentiment = sentiments.score_sentencce(token)
    features["Positive Sentiment"] = mean_sentiment[0]
    features["Negative Sentiment"] = mean_sentiment[1]
    features["sentiment"] = mean_sentiment[0] - mean_sentiment[1]
    #print(mean_sentiment[0], mean_sentiment[1])
    
    try:
        text = TextBlob(" ".join([""+i if i not in string.punctuation and not i.startswith("'") else i for i in token]).strip())
        features["Blob Polarity"] = text.sentiment.polarity
        features["Blob Subjectivity"] = text.sentiment.subjectivity
        #print (text.sentiment.polarity,text.sentiment.subjectivity )
    except:
        features["Blob Polarity"] = 0
        features["Blob Subjectivity"] = 0
        print("do nothing")
        
    
    first_half = token[0:len(token)/2]    
    mean_sentiment_half = sentiments.score_sentencce(first_half)
    features["positive Sentiment first half"] = mean_sentiment_half[0]
    features["negative Sentiment first half"] = mean_sentiment_half[1]
    features["first half sentiment"] = mean_sentiment_half[0]-mean_sentiment_half[1]
    try:
        text = TextBlob(" ".join([""+i if i not in string.punctuation and not i.startswith("'") else i for i in first_half]).strip())
        features["first half Blob Polarity"] = text.sentiment.polarity
        features["first half Blob Subjectivity"] = text.sentiment.subjectivity
        #print (text.sentiment.polarity,text.sentiment.subjectivity )
    except:
        features["first Blob Polarity"] = 0
        features["first Blob Subjectivity"] = 0
        print("do nothing")
    
    second_half = token[len(token)/2:]
    mean_sentiment_sechalf = sentiments.score_sentencce(second_half)
    features["positive Sentiment second half"] = mean_sentiment_sechalf[0]
    features["negative Sentiment second half"] = mean_sentiment_sechalf[1]
    features["second half sentiment"] = mean_sentiment_sechalf[0]-mean_sentiment_sechalf[1]
    try:
        text = TextBlob(" ".join([""+i if i not in string.punctuation and not i.startswith("'") else i for i in second_half]).strip())
        features["second half Blob Polarity"] = text.sentiment.polarity
        features["second half Blob Subjectivity"] = text.sentiment.subjectivity
        #print (text.sentiment.polarity,text.sentiment.subjectivity )
    except:
        features["second Blob Polarity"] = 0
        features["second Blob Subjectivity"] = 0
        print("do nothing")
    
        
  def pos_features(features,sentence):
    sentence_rep = replace_emoji.replace_reg(sentence)
    token = nltk.word_tokenize(sentence_rep)
    token = [ porter.stem(each.lower()) for each in token]
    pos_vector = sentiments.pos_vector(token)
    for j in range(len(pos_vector)):
        features['POS_'+str(j+1)] = pos_vector[j]
    print ("done")
    
   def topic_feature(features,sentence,topic_modeler):  
    topic_mod = topic.topic(nbtopic=200,alpha='symmetric')
    topics = topic_modeler.transform(sentence)    
    for j in range(len(topics)):
        features['Topic :'] = topics[j][1] 
    

