#!/usr/bin/env python
# coding: utf-8

import os
import nltk
from nltk.corpus import stopwords
from nltk.collocations import *
import pandas as pd


# these are required if not already downloaded
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('universal_tagset')


stopword_list = set(stopwords.words('english'))

def clean_tokenize_text(text):
    tokenized = nltk.word_tokenize(text.lower())
    words = [word for word in tokenized if (
        word.isalpha() and
        word not in [
        'cite', 'sici', 'lt', 'gt',
        ]
        and len(word) > 2
        and word not in stopword_list
    )]
    return words

def keywords(tokens):
    keywords_count = {
        'VERB': {},
        'NOUN': {},
    }
    tags = nltk.pos_tag(tokens, tagset='universal')
    for tag in tags:
        if tag[0] in stopword_list:
            continue

        if tag[1] not in ['VERB', 'NOUN']:
            continue
        if tag[0] not in keywords_count[tag[1]]:
            keywords_count[tag[1]][tag[0]] = 1
        else:
            keywords_count[tag[1]][tag[0]] += 1
    return keywords_count

def total_tokens_len(tokens):
    return len(tokens)

def total_unique_tokens_len(tokens):
    return len(set(tokens))

def bigrams(tokens, contain_verb=False):
    finder = BigramCollocationFinder.from_words(tokens, window_size=5)
    finder.apply_freq_filter(3)
    return finder.ngram_fd.items()

def trigrams(tokens, contain_verb=False):
    finder = TrigramCollocationFinder.from_words(tokens, window_size=5)
    finder.apply_freq_filter(3)
    return finder.ngram_fd.items()


offset = 0
max_citations = 10000
total_runs = 300
prev_total_tokens = 0
prev_unique_tokens = 0


def save_keywords(offset, v, k):
    words = keywords(v)
    df = pd.DataFrame(words['VERB'].items(), columns =['verbs', 'freq'])
    df[['verbs', 'freq']].to_csv(f'./data/{k}_{offset}_verbs.csv')

    df = pd.DataFrame(words['NOUN'].items(), columns =['nouns', 'freq'])
    df[['nouns', 'freq']].to_csv(f'./data/{k}_{offset}_nouns.csv')

def save_bigrams(offset, v, k):
    bg = bigrams(v)
    df = pd.DataFrame(bg, columns =['bigrams', 'freq'])
    df['bigrams'] = df['bigrams'].astype(str)
    df[['bigrams', 'freq']].to_csv(f'./data/{k}_{offset}_bigrams.csv')

def save_trigrams(offset, v, k):
    tg = trigrams(v)
    df = pd.DataFrame(tg, columns =['trigrams', 'freq'])
    df['trigrams'] = df['trigrams'].astype(str)
    df[['trigrams', 'freq']].to_csv(f'./data/{k}_{offset}_trigrams.csv')

print('Starting collocate and keyword generation')
for i in range(total_runs):
    for type in ['supporting', 'contradicting', 'mentioning']:
        print(f"\nFor {type}, Running {max_citations} at offset: {offset * max_citations}\n")
        df = pd.read_sql(os.environ['SQL_STRING'], os.environ['SQL_ACCESS'])
        tokens = []
        print(f"tokenizing text for offset {offset * max_citations}")
        for i, row in df.iterrows():
            sentence = row['text']
            words = clean_tokenize_text(sentence)
            tokens.extend(words)
        del df

        print(f'Compiling citation data for {type}_{offset} citations')
        print(f'Saving data for {type}')
        df = pd.DataFrame([
            [total_tokens_len(tokens) + prev_total_tokens]
        ] , columns =['total_tokens'])
        df.to_csv(f'./data/{type}_metadata.csv')
        prev_total_tokens += total_tokens_len(tokens)
        prev_unique_tokens += total_unique_tokens_len(tokens)

        print('Saving Verb and Noun Freq')
        save_keywords(offset, tokens, type)

        print('Saving Bigrams')
        save_bigrams(offset, tokens, type)

        print('Saving Trigrams')
        save_trigrams(offset, tokens, type)

    offset += 1


