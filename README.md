# citation-function-corpus-lingustics
Analysis of significant terms and collocates of citations by citation function (classified by scite.ai) presented in the paper: Citations are not opinions: a corpus linguistics approach to understanding how citations are made (manuscript not yet available - available on request)
## Description

This repo contains the data and analysis for studying the corpus lingustics of citations by their citation function (a classification whether the citation is supporting, disputing/contradicting, mentioning). The method taken is to generate terms (verbs and nouns) and collocates (bigrams and trigrams).

`./generates_collocates.py` - generates the terms and collocates from the scite.ai citation statement dataset resulting in the `./data` folder which are a count of structures by citation classification.

`./analyize_collocates.py` - analyzes the `./data` folder to understand lingustic charateristics of the subcorpora and unique terms and phrases based on their loglikelihood.

### Analysis performed

The analysis present in `./analysis` performed is best described in the `Analyze Collocates` notebook. Generally though we have a feature, such as trigrams, a statistical test such as supporting citations taking the greatest log likelihood, and a lingustic analysis such as POS tag distrubtions, sentiment, subjectivity and more.
## Reproduction and Setup

Dependencies are managed by poetry. run `poetry install` once poetry is installed and everything should be available. You can run the notebooks with `poetry run jupyter notebook`

Install the dependencies found in the notebooks and scripts. This will include downloading some nltk files.

To generate the ./data files you will need access to the scite.ai citation corpus which may be available on request (contact hi@scite.ai). Using that access you can run `generate_collocates.py`

A small sample to generate these is available in citations_sample.csv and can be generated and explored using the `Generate Collocates and Keywords Notebook`.

To generate the ./analysis files simply run `analyze_collocates.py`. Again only a small sample of files is included for brevity because the sample size in the original study is 6 million citations and is too big to include here (that data is available on request).

## Exploration

To explore the data use some of the cells in `Analyze Collocates` notebook to explore the terms and ngrams in `./data`
