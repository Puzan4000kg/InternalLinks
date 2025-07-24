#!/bin/bash
set -e

# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download ru_core_news_sm

# Download NLTK data
python -m nltk.downloader punkt stopwords averaged_perceptron_tagger

# Verify installations
python -c "import spacy; nlp=spacy.load('en_core_web_sm'); print('spaCy model loaded successfully')"
python -c "import nltk; nltk.data.find('tokenizers/punkt'); print('NLTK data loaded successfully')" 