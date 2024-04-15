[![Dependencies](https://img.shields.io/badge/Dependencies-NLTK%2C%20scikit--learn%2C%20pandas-blue)](https://pypi.org/)
https://img.shields.io/badge/code-Python-red?style=flat-square
https://img.shields.io/badge/dependencies-NLTK-blue
https://img.shields.io/badge/dependencies-scikitlearn-darkblue?style=plastic
https://img.shields.io/badge/dependencies-pandas-lightblue?style=plastic
https://img.shields.io/badge/dependencies-networx-white?style=plastic



This is an automatic extractive text summarization algorithm.

The working principle of extractive text summarization idea is that the model generates summaries using only the words that are already contained in the original text. Compared to the abstractive text summarization algorithms, these are easier to implement, do not necessarily require network training, but are less accurate and useful.

Extractive document summarization algorithms rank the pre-processed sentences in the original text depending on some selected features and produce a summary using solely these ranked sentences. 
The main algorithm that is followed throughout this project is the TextRank algorithm which is a graph-based summarization algorithm inspired by PageRank algorithm. Sentences are represented as nodes where connections between them are the edges. 
After pre-processing of the text documents, features are extracted and they are put into a cosine similarity matrix which is then used to produce the graphs and finally rank the sentences.

##Project outline

The main dependencies are; NLTK, which is used mainly by taking advantage of tokenizers and lemmatizers in pre-processing step, and scikitlearn, which is useful in feature extraction.

Pre-processing step includes; special character and punctuation removal, case conversion, tokenization, stop-word removal, and lemmatization. 

After these, feature extraction, whose sub-sections are; N-gram bag of words, word frequency vectorizer, and TF-IDF vectorizer.
Finally, sentence ranks are calculated using PageRank algorithm and summaries are generated for the News category of Brown corpus.

##Data

Brown and Reuters corpora are used via NLTK library. 
Brown corpus is the main set that the model uses to generate summaries and Reuters corpus is used only for trial purposes.
