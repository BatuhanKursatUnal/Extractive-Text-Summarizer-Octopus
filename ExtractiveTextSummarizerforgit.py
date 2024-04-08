#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 18:44:35 2023

@author: batuhanmac
"""

# Automatic Extractive Text Summarization Algorithm
# Written by Batuhan Kursat Unal, University of Bologna, September 2023

# Importing necessary libraries and modules

# NLTK Library Dependencies (mostly for pre-processing of raw text)
import nltk
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.stem import WordNetLemmatizer

import numpy as np
import random as rn
import pandas as pd
import string

# Scikitlearn library (mainly for feature extraction)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Downloading and importing the corpora

from nltk.corpus import brown
from nltk.corpus import reuters

raw_brown_news = brown.raw(categories = 'news')

# Pre-processing of the Data


# Special character and punctuation removal
# I create a set of punctuations to be removed specially, because I want to keep words like "I'll" intact.
my_punctuation = string.punctuation.replace("'", "") + "``" + "''" + "--"


# For tokenized sentences

def special_ch_punc_removal_sent(ch_punc_in):
    '''
    Parameters
    ----------
    ch_punc_in : Whole document

    Returns
    -------
    ch_punc_out : Output, special characters and punctuations are removed.
    
    '''
    
    cleaned_sentence = []
    ch_punc_out = []
    words = ch_punc_in.split()
    
    cleaned_sentence = [word for word in words if word not in my_punctuation]
 
    ch_punc_out = ' '.join(cleaned_sentence)
    return ch_punc_out





# For tokenized words

def special_ch_punc_removal(clean_tokens):
    
    clean_tokens = [token for token in clean_tokens if token not in my_punctuation]
    
    return clean_tokens


# Case conversion

# For tokenized words

def case_converter(clean_tokens):
    
    clean_tokens = [token.lower() for token in clean_tokens]
    
    return clean_tokens

# For tokenized sentences

def case_converter_sent(sent_tokens_in):
    sent_tokens_out = [string.lower() for string in sent_tokens_in]
    
    return sent_tokens_out


# Stop word removal

stopwords = nltk.corpus.stopwords.words('english')

# For tokenized words

def stop_word_removal(clean_tokens):
    
    clean_tokens = [token for token in clean_tokens if token not in stopwords]
    
    return clean_tokens

# For tokenized sentences

def stop_word_removal_sent(sent_tokens_in):
    
    sent_tokens_out = [string for string in sent_tokens_in if string not in stopwords]
    
    return sent_tokens_out


# Creating Word and Sentence Tokenizers

# This tokenizer function employs NLTK Regexp word tokenizer. 
# Further processing is performed in the function to get clean and appropriate results.

def tokenize_text(ch_text):
    '''
    Parameters
    ----------
    ch_text : The chosen text for tokenization

    Returns
    -------
    clean_tokens : Tokenized words
    
    '''

    # Word tokenizer
    GAP_PATTERN = r'\s+'
    regex_wt = nltk.RegexpTokenizer(pattern=GAP_PATTERN, gaps=True)
    word_tokens_cat_regexp = regex_wt.tokenize(ch_text)     
    
    # Since the tokenized words also contain POS tags with them and the aim here is to produce readable summaries, 
    # I omit these tags.
    clean_tokens = [word.split("/")[0].strip() for word in word_tokens_cat_regexp]          

    return clean_tokens

    
# Sentence Tokenization
# Another Way of Tokenizing Sentences

def tokenize_text_sent_v2(in_text_words):
    '''
    Parameters
    ----------
    in_text_words : Input must be employing the attribute .words() of NLTK library

    Returns
    -------
    sent_tokens12 : Tokenized sentences
    
    '''
    

    sent_tokens12 = sent_tokenize(in_text_words, language= 'english')
    
    return sent_tokens12

# Demonstration of Sentence Tokenizer Version2
in_text_words_01 = brown.words('ca06') # Arbitrary choice of a document
sent_tokens01 = tokenize_text_sent_v2(in_text_words_01)
sent_tokens01


# Lemmatization

wnl = WordNetLemmatizer()

def lemmatizer(clean_tokens):
    
    tokens_pos_tags = nltk.pos_tag(clean_tokens)
    
    # Mapping NLTK part-of-speech tags to WordNetLemmatizer tags

    pos_tags_map = {'NN' : 'n', 'NNP' : 'n', 'NNPS' : 'n', 'NNS' : 'n', 'JJ' : 'a', 'JJR' : 'a', 'JJS' : 'a',
                    'RB' : 'r', 'RBR' : 'r', 'RBS' : 'r', 'VB' : 'v', 'VBD' : 'v', 'VBG' : 'v', 'VBN' : 'v',
                    'VBP' : 'v', 'VBZ' : 'v', 'CC' : 'n', 'CD' : 'n', 'DT' : 'n', 'EX' : 'n', 'FW' : 'n',
                    'IN' : 'n', 'LS' : 'n', 'MD' : 'n', 'PDT' : 'n', 'POS' : 'n', 'PRP' : 'n', 'PRP$' : 'n',
                    'RP' : 'n', 'SYM' : 'n', 'WRB' : 'n', 'WP' : 'n', 'WP$' : 'n'}

    pos_tags = []
    for pos in tokens_pos_tags:
        pos_tags += [pos[1]]
                 
    poses = []
    for pos in pos_tags:
        if pos in pos_tags_map:
            poses += pos_tags_map[pos]
        else:
            poses += 'n'
        
    lemmas = [wnl.lemmatize(token, pos) for token, pos in zip(clean_tokens,poses)]
    
    return lemmas, pos_tags, poses



# Pre-processing all of the texts in the news category

# In this data frame, you can find tokenized words, sentences and lemmas for each of the texts in news category of Brown corpus.
# This can easily be extended into larger datasets, i.e. texts from other categories of the corpus.

brown_news_allids = brown.fileids(categories = 'news')
df = pd.DataFrame(columns = ["Tokenized Words", "Tokenized Sentences", "Lemmas"])

count = 0

for fileid in brown_news_allids:
    
    count += 1
    
    # For word tokenization and pre-processing to get "clean_tokens3"
    in_text_doc = ' '.join(brown.words(fileids = fileid))
    
    clean_tokens = tokenize_text()
    clean_tokens1 = special_ch_punc_removal(clean_tokens)
    clean_tokens2 = case_converter(clean_tokens1)
    clean_tokens3 = stop_word_removal(clean_tokens2)
    
    # For sentence tokenization and pre-processing to get "sent_tokens2"
    
    sent_tokens = tokenize_text_sent(clean_tokens)
    sent_tokens1 = case_converter_sent(sent_tokens)
    sent_tokens2 = stop_word_removal_sent(sent_tokens1)
    
    # Lemmatization
    
    lemmas1, pos_tags, poses = lemmatizer(clean_tokens3)
    
    
    df.loc[count] = [clean_tokens3, sent_tokens2, lemmas1]
    

# Feature Extraction

# N-gram Bag of Words 

def ngram_bow_vec(sentences):
    
    '''
    Parameters
    ----------
    sentences : The input is the document that is chosen by hand. It must be indicated using the row and column
    numbers through df.iloc[i]][j]

    Returns
    -------
    bow_arr : N-gram bag of words matrix representing each sentence as its rows and each unigram or bigram 
    as its columns

    '''
    
    bow_vectorizer = CountVectorizer(binary = True, ngram_range = (1, 2))
    bow_vec = bow_vectorizer.fit_transform(sentences)
    bow_arr = bow_vec.toarray()
    
    return bow_arr


# Word Frequency Vectorizer

def wordfreq_vec(sentences):
    
    '''
    Parameters
    ----------
    sentences : The input is the document that is chosen by hand. It must be indicated using the row and column
    numbers through df.iloc[i]][j]

    Returns
    -------
    word_freq_arr : Word to sentence embedding of word frequency in each sentence.
    
    '''
    
    freq_vectorizer = CountVectorizer(binary = False, min_df = 1, ngram_range = (1, 1))
    word_freq = freq_vectorizer.fit_transform(sentences)
    word_freq_arr = word_freq.toarray()
    
    return word_freq_arr

# Trying Word Frequency Feature function for the sentences in document 0
in_sent_0 = df.iloc[0][1]
wordfreq_arr_0 = wordfreq_vec(in_sent_0)


# Putting all the documents in a dictionary to be able to produce summaries quickly
doc_dict = {}
for i in range(len(df)):
    chosen_doc = df.iloc[i][1]
    var_name = f'in_sent_{i}'
    el_name = f'chosen_doc {i}'
    
    doc_dict[var_name] = el_name
        

# TF-IDF Vectorizer
# Term Frequency - Inverse Document Frequency

def tfidf(sentences):
    
    '''
    Parameters
    ----------
    sentences : The input is the document that is chosen by hand. It must be indicated using the row and column
    numbers through df.iloc[i]][j]

    Returns
    -------
    tfidf_arr : Word to sentence embedding of tfidf scores.
    
    '''
    
    tfidf_vec = TfidfVectorizer(min_df = 2, max_df = 0.5, ngram_range=(1, 2))
    tfidf_scores = tfidf_vec.fit_transform(sentences)
    tfidf_arr = tfidf_scores.toarray()
    
    return tfidf_arr


# Trying TF-IDF function for the sentences in document 0
tfidf_arr_0 = tfidf(in_sent_0)


# Sentence Position
sent_position = [i / len(sent_tokens2) for i in range(len(sent_tokens2))]

# Combine TF-IDF and Word Frequency Features for Sentences
combined_features = np.concatenate((tfidf_arr_0, wordfreq_arr_0), axis=1)

# Cosine Similarity Matrices

# Cosine Similarity Matrix for TF-IDF Scores
# It will give us the amount of similarity of each sentence with the others in the document. In the case of document 0
# there are 92 sentences so it is a 92x92 matrix
def cos_sim(features):
    
    cos_sim = cosine_similarity(features)
    
    return cos_sim

# Cosine similarity matrix for combined features
cossim_arr_0_cf = cos_sim(combined_features)



# Trying the Cosine Similarity function for sentences in document 0 and tfidf feature
cossim_arr_0_tfidf = cos_sim(tfidf_arr_0)

# Trying the Cosine Similarity Matrix using Word Frequency Feature for sentences in document 0
cossim_arr_0_wordfreq = cos_sim(wordfreq_arr_0)

# Using tfidf score

# Creating a graph out of the cosine similarity matrix using tfidf score
cossim_graph_0_tfidf = nx.from_numpy_array(cossim_arr_0_tfidf)
scores_0_tfidf = nx.pagerank(cossim_graph_0_tfidf)

# Determining sentence ranks by calculating their scores using PageRank algorithm
ranked_sentences_0_tfidf = sorted(((scores_0_tfidf[i], s) for i, s in enumerate(in_sent_0)), reverse = True)

# Generate the summary
summary1 = ''
for i in range(5):
    summary1 += ranked_sentences_0_tfidf[i][1]
    
print('The first summary is as follows: \n', summary1)


# Using word frequency feature

# Creating a graph out of the cosine similarity matrix using word frequency feature
cossim_graph_0_wordfreq = nx.from_numpy_array(cossim_arr_0_wordfreq)
scores_0_wordfreq = nx.pagerank(cossim_graph_0_wordfreq)

# Determining sentence ranks by calculating their scores using PageRank algorithm
ranked_sentences_0_wordfreq = sorted(((scores_0_wordfreq[i], s) for i, s in enumerate(in_sent_0)), reverse = True)

# Generate the summary
summary2 = ''
for i in range(5):
    summary2 += ranked_sentences_0_wordfreq[i][1]
    
print('\n The second summary is as follows: \n', summary2, '\n')


# Using combined features

# Creating a graph out of the cosine similarity matrix using combined features
cossim_graph_0_cf = nx.from_numpy_array(cossim_arr_0_cf)
scores_0_cf = nx.pagerank(cossim_graph_0_cf)

# Determining sentence ranks by calculating their scores using PageRank algorithm
ranked_sentences_0_cf = sorted(((scores_0_cf[i], s) for i, s in enumerate(in_sent_0)), reverse = True)

# Generate the summary
summary3 = ''
for i in range(5):
    summary3 += ranked_sentences_0_cf[i][1]
    
print('\n The third summary is as follows: \n', summary3, '\n')


# Iterative method to generate and store all the summaries of documents in the news category of Brown corpus
all_summaries = []

for i in range(len(df)):

    in_sent_i = df.iloc[i][1] # Each document is chosen and stored iteratively
    
    # Calculate features (TF-IDF, word frequency, and combined)
    tfidf_arr_i = tfidf(in_sent_i)
    wordfreq_arr_i = wordfreq_vec(in_sent_i)
    bow_arr_i = ngram_bow_vec(in_sent_i)
    
    combined_features_i = np.concatenate((tfidf_arr_i, wordfreq_arr_i, bow_arr_i), axis=1)
    
    # Calculating Cosine Similarity Matrix
    cossim_i = cos_sim(combined_features_i)
    
    # Creating a graph and calculating scores using PageRank Algorithm
    cossim_graph_i = nx.from_numpy_array(cossim_i)
    scores_i = nx.pagerank(cossim_graph_i)
    
    # Determine sentence ranks
    ranked_sentences_i = sorted(((scores_i[j], s) for j, s in enumerate(in_sent_i)), reverse=True)
    
    # Generate the Summaries
    summary_i = ''
    for j in range(5):  # The number of sentences in the summary could be adjusted as needed
        summary_i += ranked_sentences_i[j][1]
    
    # Append the summary to the list of all summaries
    all_summaries.append(summary_i)

# Print all the summaries
for i, summary in enumerate(all_summaries):
    print(f'Summary for document {i}:\n{summary}\n')

