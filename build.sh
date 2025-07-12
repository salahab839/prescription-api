#!/usr/bin/env bash
# exit on error
set -e

pip install -r requirements.txt

python -m spacy download fr_core_news_sm
