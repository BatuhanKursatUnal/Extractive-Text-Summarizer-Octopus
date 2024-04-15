https://img.shields.io/badge/dependencies-NLTK-brightgreen?style=plastic&color=rgba

This is an automatic extractive text summarization algorithm.

The working principle of extractive text summarization idea is that the model generates summaries using only the words that are already contained in the original text. Compared to the abstractive text summarization algorithms, these are easier to implement, do not necessarily require network training, but are less accurate and useful.

Extractive document summarization algorithms rank the pre-processed sentences in the original text depending on some selected features and produce a summary using solely these ranked sentences. 

********The project outline is as follows:********
The main dependencies are; NLTK, which is used mainly by taking advantage of tokenizers and lemmatizers in pre-processing step, and scikitlearn, which is useful in feature extraction.

Brown and Reuters corpora are used via NLTK library. Brown corpus is the main set that the model uses to generate summaries and Reuters corpus is used only for trial purposes.

Pre-processing step includes; special character and punctuation removal, case conversion, tokenization, stop-word removal, and lemmatization. 

After these, feature extraction, whose sub-sections are; N-gram bag of words, word frequency vectorizer, and TF-IDF vectorizer.
Finally, sentence ranks are calculated using PageRank algorithm and summaries are generated for the News category of Brown corpus.
