#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.collocations import *
import pandas as pd
import csv
import math
from matplotlib import pyplot
import seaborn as sns
from textblob import TextBlob
from ast import literal_eval as make_tuple
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

def loglikelihood(word_freq_c1, word_freq_c2, total_tokens_c1, total_tokens_c2):
    if word_freq_c1 == 0 and word_freq_c2 == 0:
        return 0

    total_tokens_c12 = total_tokens_c1 + total_tokens_c2
    relative_freq_c1 = (total_tokens_c1 * (word_freq_c1 + word_freq_c2)) / total_tokens_c12
    relative_freq_c2 = (total_tokens_c2 * (word_freq_c1 + word_freq_c2)) / total_tokens_c12

    log1 = 0
    if word_freq_c1 > 0:
        log1 = math.log(word_freq_c1 / relative_freq_c1)

    log2 = 0
    if word_freq_c2 > 0:
        log2 = math.log(word_freq_c2 / relative_freq_c2)
    return 2 * (
        word_freq_c1 * (log1) +
        word_freq_c2 * (log2)
    )


def join_batched_tables(citation_type, batches):
    analysise = ['verbs', 'nouns', 'bigrams', 'trigrams']
    for analysis in analysise:
        prev_df = None
        for batch in range(batches):
            print(batch, analysis)
            df = pd.read_csv(f"./data/{citation_type}_{batch}_{analysis}.csv")
            df = df[[analysis, 'freq']]
            if batch != 0:
                df = pd.concat([df, prev_df]).groupby(by=analysis)[analysis, 'freq'].sum().reset_index()
            prev_df = df.copy()

        df.to_csv(f'./analysis/{citation_type}_{analysis}.csv')


join_batched_tables('supporting', 10)
join_batched_tables('contradicting', 10)
join_batched_tables('mentioning', 10)


def generate_comparison_tables():
    analysise = ['bigrams', 'trigrams', 'verbs', 'nouns']
    for analysis in analysise:
        supporting_df = pd.read_csv(f'./analysis/supporting_{analysis}.csv').rename(columns={"freq": "supporting_freq"})
        supporting_df = supporting_df[supporting_df['supporting_freq'] > 10]
        disputing_df = pd.read_csv(f'./analysis/contradicting_{analysis}.csv').rename(columns={"freq": "disputing_freq"})
        disputing_df = disputing_df[disputing_df['disputing_freq'] > 10]
        mentioning_df = pd.read_csv(f'./analysis/mentioning_{analysis}.csv').rename(columns={"freq": "mentioning_freq"})
        mentioning_df = mentioning_df[mentioning_df['mentioning_freq'] > 10]
        supporting_metadata_df = pd.read_csv(f'./data/supporting_metadata.csv')
        disputing_metadata_df = pd.read_csv(f'./data/contradicting_metadata.csv')
        mentioning_metadata_df = pd.read_csv(f'./data/mentioning_metadata.csv')

        result_a = pd.merge(supporting_df[[analysis, 'supporting_freq']], disputing_df[[analysis, 'disputing_freq']], how='outer', on=analysis).fillna(0)
        result_b = pd.merge(result_a, mentioning_df[[analysis, 'mentioning_freq']], how='outer', on=analysis).fillna(0)

        print('computing supporting disputing loglikelihood')
        result_b['supporting_disputing_loglikelihood'] = result_b.apply(
            lambda x:
            loglikelihood(x['supporting_freq'], x['disputing_freq'], supporting_metadata_df['total_tokens'][0], disputing_metadata_df['total_tokens'][0])
        ,axis=1)
        print('computing supporting  mentioning loglikelihood')
        result_b['supporting_mentioning_loglikelihood'] = result_b.apply(
            lambda x:
            loglikelihood(x['supporting_freq'], x['mentioning_freq'], supporting_metadata_df['total_tokens'][0], mentioning_metadata_df['total_tokens'][0])
        ,axis=1)
        print('computing disputing  mentioning loglikelihood')
        result_b['disputing_mentioning_loglikelihood'] = result_b.apply(
            lambda x:
            loglikelihood(x['disputing_freq'], x['mentioning_freq'], disputing_metadata_df['total_tokens'][0], mentioning_metadata_df['total_tokens'][0])
        ,axis=1)
        result_b.to_csv(f"./analysis/{analysis}_comparison_table.csv")


generate_comparison_tables()


from ast import literal_eval as make_tuple
stopword_list = stopwords.words('english')

# it should be greater than both I think
def disputing_greater_flt(x):
    return x['disputing_freq'] >=  x['supporting_freq'] and x['disputing_freq'] >=  x['mentioning_freq']

def supporting_greater_flt(x):
    return x['supporting_freq'] >=  x['disputing_freq'] and x['supporting_freq'] >=  x['disputing_freq']

def mentioning_greater_flt(x):
    return x['mentioning_freq'] >=  x['disputing_freq'] and x['mentioning_freq'] >=  x['supporting_freq']


def common_flt(x):
    return x['supporting_disputing_loglikelihood'] <= 3.84 and x['supporting_mentioning_loglikelihood'] <= 3.84


# What are the top 100 highest liklihood across tables
# log likliehood filters

# significance table:
# 95th percentile; 5% level; p < 0.05; critical value = 3.84
# 99th percentile; 1% level; p < 0.01; critical value = 6.63
# 99.9th percentile; 0.1% level; p < 0.001; critical value = 10.83
# 99.99th percentile; 0.01% level; p < 0.0001; critical value = 15.13
def top_likelihood(significance=3.84):
    for structure in [ 'verbs', 'nouns', 'bigrams', 'trigrams']:
        df = pd.read_csv(f"./analysis/{structure}_comparison_table.csv")
        analysize = [
            {
                'name': 'supporting_greater',
                'likelihood': 'supporting_disputing_loglikelihood',
                'filter': supporting_greater_flt,
            },
            {
                'name': 'disputing_greater',
                'likelihood': 'supporting_disputing_loglikelihood',
                'filter': disputing_greater_flt,
            },
            {
                'name': 'mentioning_greater',
                'likelihood': 'supporting_mentioning_loglikelihood',
                'filter': mentioning_greater_flt,
            },
                      {
                'name': 'common',
                'likelihood': 'mentioning_freq',
                'filter': common_flt,
            },
        ]
        for analysis in analysize:
            af = df[df[analysis['likelihood']] >= significance]
            af = af[af.apply(analysis['filter'], axis=1)].sort_values(analysis['likelihood'])
            af.to_csv(f"./analysis/{structure}_{analysis['name']}.csv", index=False)

top_likelihood()

def make_pos(x, pos):
    tup = make_tuple(x)
    tokenized = nltk.word_tokenize(' '.join(tup))
    tags = nltk.pos_tag(tokenized)
    tagstr = ''
    for tag in tags:
        tagstr += tag[1] + ' '
    return tagstr

def find_pos(x, pos):
    tup = make_tuple(x)
    tokenized = nltk.word_tokenize(' '.join(tup))
    tags = nltk.pos_tag(tokenized)
    for tag in tags:
        if tag[1] == pos:
            return True
    return False

def sentiment(x):
    tup = make_tuple(x)
    sent = sia.polarity_scores(' '.join(tup))
    return sent['compound']

def discourse_markers(x):
    tup = make_tuple(x)
    sent = ' '.join(tup)
    for dm in DISCOURSE_MARKERS:
        if dm in sent:
            return True
    return False


# In[ ]:


interesting_pos = ['RB','RBR', 'RBS', 'JJ', 'JJR', 'JJS', 'IN', 'CC']
POS = {
    "CC": "coordinating conjunctions",
    "IN": "prepositions and subordinating conjunctions",
    "JJ": "adjectives",
    "JJR": "adjectives, comparative",
    "JJS": "adjectives, superlative",
    "RB": "adverbs",
    "RBR": "adverbs, comparative",
    "RBS": "adverbs, superlative"
}

analysize = ['bigrams', 'trigrams']
for analysis in analysize:
    for pos in interesting_pos:
        supporting_counts = []
        disputing_counts = []
        mentioning_counts = []
        df = pd.read_csv(f"./analysis/{analysis}_supporting_greater.csv")
        of = df[df.apply(lambda x: find_pos(x[analysis], pos), axis=1)]
        of.to_csv(f"./analysis/{analysis}_{pos}_supporting_greater.csv")
        supporting_counts.append(len(of) / len(df))
        print(f"supporting - {analysis} - {pos}: {len(of) / len(df)} ")

        df = pd.read_csv(f"./analysis/{analysis}_disputing_greater.csv")
        of = df[df.apply(lambda x: find_pos(x[analysis], pos), axis=1)]
        of.to_csv(f"./analysis/{analysis}_{pos}_disputing_greater.csv")
        disputing_counts.append(len(of) / len(df))
        print(f"disputing - {analysis} - {pos}: {len(of) / len(df)} ")


        df = pd.read_csv(f"./analysis/{analysis}_mentioning_greater.csv") # it should be mentioning greater than both
        of = df[df.apply(lambda x: find_pos(x[analysis], pos), axis=1)]
        of.to_csv(f"./analysis/{analysis}_{pos}_mentioning_greater.csv")
        mentioning_counts.append(len(of) / len(df))
        print(f"mentioning - {analysis} - {pos}: {len(of) / len(df)} ")


        df = pd.DataFrame({
            'Factor': pos,
            'Supporting citations': supporting_counts,
            'Disputing citations': disputing_counts,
            'Mentioning citations': mentioning_counts,
        })
        fig, ax1 = pyplot.subplots(figsize=(10, 10))
        tidy = df.melt(id_vars='Factor').rename(columns=str.title)
        sns.barplot(x='Factor', y='Value', hue='Variable', data=tidy, ax=ax1)
        ax1.set(title=f"Frequency of {POS[pos]} ({pos}) in {analysis} (%)", xlabel=f"Part of speech: {POS[pos]}", ylabel=f"Frequency of occurance(%)")
        vals = ax1.get_yticks()
        ax1.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
        sns.despine(fig)
        pyplot.savefig(f'./analysis/{analysis}_{pos}.png')
        pyplot.plot()

# bar and counts of discourse markers
# plot sentiments and correlates


# In[ ]:


analysize = ['bigrams', 'trigrams']
functions = [
    {
        'name': 'Supporting',
        'type': 'supporting_greater',
        'function': 'supporting_disputing_loglikelihood'
    },
    {
        'name': 'Disputing',
        'type': 'disputing_greater',
        'function': 'supporting_disputing_loglikelihood'
    },
    {
        'name': 'Mentioning',
        'type': 'mentioning_greater',
        'function': 'supporting_mentioning_loglikelihood'
    }
]
for analysis in analysize:
    for function in functions:
        cite_type = function["type"]
        df = pd.read_csv(f"./analysis/{analysis}_{cite_type}.csv")
        df['sent'] = df.apply(lambda x: sentiment(x[analysis]), axis=1)
        print(function['name'])
        print(df[['sent', function['function']]].corr(method = 'pearson'))
        df.sort_values('sent').to_csv(f"./analysis/{analysis}_{cite_type}_sent.csv")

        fig, ax1 = pyplot.subplots(figsize=(10, 10))
        sns.scatterplot(data=df, x='sent', y=function['function'], ax=ax1)
        ax1.set(title=f"{function['name']} {analysis} sentiment versus log likelihood", ylabel="Log Likelihood", xlabel=f"Sentiment (-1 negative, 0 neutral, 1 positive)")
        vals = ax1.get_yticks()
        sns.despine(fig)
        cite_name = function['type']
        pyplot.savefig(f'./analysis/{analysis}_{cite_name}_sent_scatter.png')
        pyplot.plot()


def subjectivity(x):
    tup = make_tuple(x)
    sent = TextBlob(' '.join(tup)).sentiment
    return sent[1]


analysize = ['bigrams', 'trigrams']
functions = [
    {
        'name': 'Supporting',
        'type': 'supporting_greater',
        'function': 'supporting_disputing_loglikelihood'
    },
    {
        'name': 'Disputing',
        'type': 'disputing_greater',
        'function': 'supporting_disputing_loglikelihood'
    },
    {
        'name': 'Mentioning',
        'type': 'mentioning_greater',
        'function': 'supporting_mentioning_loglikelihood'
    }
]
for analysis in analysize:
    for function in functions:
        cite_type = function["type"]
        df = pd.read_csv(f"./analysis/{analysis}_{cite_type}.csv")
        df['subj'] = df.apply(lambda x: subjectivity(x[analysis]), axis=1)
        print(function['name'])
        print(df[['subj', function['function']]].corr(method = 'pearson'))
        df.sort_values('subj').to_csv(f"./analysis/{analysis}_{cite_type}_subj.csv")

        fig, ax1 = pyplot.subplots(figsize=(10, 10))
        sns.histplot(data=df, x='subj', ax=ax1)
        ax1.set(title=f"{function['name']} {analysis} subjectivity", xlabel=f"Count of {analysis}", ylabel=f"Subjectivity (0.0 very objective, 1.0 very subjective)")
        vals = ax1.get_yticks()
        sns.despine(fig)
        cite_name = function['type']
        pyplot.savefig(f'./analysis/{analysis}_{cite_name}_hist_plot.png')
        pyplot.plot()

        fig, ax1 = pyplot.subplots(figsize=(10, 10))
        sns.scatterplot(data=df, x='subj', y=function['function'], ax=ax1)
        ax1.set(title=f"{function['name']} {analysis} subjectivity versus log likelihood", ylabel="Log Likelihood", xlabel=f"Subjectivity (0.0 very objective, 1.0 very subjective)")
        vals = ax1.get_yticks()
        sns.despine(fig)
        cite_name = function['type']
        pyplot.savefig(f'./analysis/{analysis}_{cite_name}_subj_scatter.png')
        pyplot.plot()

