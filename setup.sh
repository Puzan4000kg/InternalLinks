#!/bin/bash
python -m spacy download en_core_web_sm
python -m spacy download ru_core_news_sm
python -m nltk.downloader punkt stopwords averaged_perceptron_tagger 