import pandas as pd
import os
import spacy
import random
import re
import numpy as np
import string
import nltk
from nltk.stem import WordNetLemmatizer
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import names 
import sys
from unicodedata import category
from cleantext import clean
punctuations =  [chr(i) for i in range(sys.maxunicode) if category(chr(i)).startswith("P")]
nltk.download('names')
male_names = names.words('male.txt')
female_names = names.words('female.txt')

all_names = []
for name in male_names:
    all_names.append(name.lower())
for name in female_names:
    all_names.append(name.lower())

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
nltk.download('wordnet')
nltk.download('omw-1.4')
wordnet_lemmatizer = WordNetLemmatizer()
nltk.download('punkt')


def tokenizer(text):
    return word_tokenize(text)[:500]

def remove_punctuation(tokens):
    no_punctiation = [token for token in tokens if token not in punctuations]
    return no_punctiation

def remove_stopwords(tokens):
    no_stopwords= [token for token in tokens if token not in stopwords]
    return no_stopwords

def standardize_names(tokens):
    standardized = [token if token not in all_names else '< PERSON >' for token in tokens]
    return standardized

def lemmatizer(tokens): # i prefer lemmatization over stemming since it preserves the meaning more
    lemmatized = [wordnet_lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized

def tokens_to_text(tokens):
    text = "".join([token+" " for token in tokens])[:-1]
    return text

def general_cleaning(text):
    return clean(text,
    fix_unicode=True,               # fix various unicode errors
    to_ascii=True,                  # transliterate to closest ASCII representation
    lower=True,                     # lowercase text
    no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
    no_urls=True,                  # replace all URLs with a special token
    no_emails=True,                # replace all email addresses with a special token
    no_phone_numbers=True,         # replace all phone numbers with a special token
    no_numbers=True,               # replace all numbers with a special token
    no_digits=True,                # replace all digits with a special token
    no_currency_symbols=True,      # replace all currency symbols with a special token
    no_punct=True,                 # remove punctuations
    replace_with_punct="",          # instead of removing punctuations you may replace them
    replace_with_url="<URL>",
    replace_with_email="<EMAIL>",
    replace_with_phone_number="<PHONE>",
    replace_with_number="<NUMBER>",
    replace_with_digit="0",
    replace_with_currency_symbol="<CUR>",
    lang="en"                       # set to 'de' for German special handling
)
def preprocess_and_write(df, path):

    df['title_text'] = df['title_text'].apply(lambda x:general_cleaning(x))
    print("sequence has been cleaned with clean-text library.")
    df['title_text'] = df['title_text'].apply(lambda x:tokenizer(x))
    print("sequence has been tokenized.")
    df['title_text']= df['title_text'].apply(lambda x:remove_stopwords(x))
    print("stop words has been removed.")
    df['title_text']= df['title_text'].apply(lambda x:standardize_names(x))
    print("names has been standardized.")
    df['title_text']= df['title_text'].apply(lambda x:lemmatizer(x))
    print("sequence has been lemmatized.")
    df['title_text']= df['title_text'].apply(lambda x:tokens_to_text(x))
    print("sequence has been converted to text again.")
    df.to_csv(path, index=False)
    print("preprocessed train/dev dataset is ready.")

data_folder = "/app/data"

train = pd.read_csv(os.path.join(data_folder,"task3_engslish_training.csv"), on_bad_lines='skip')

train = train.drop(columns=['public_id'])
train.columns = ['text', 'title', 'label']

train.loc[train["label"] == "TRUE", "label"] = "truth"
train.loc[train["label"] == "FALSE", "label"] = "fake"
train.loc[train["label"] == "partially false", "label"] = "partial"
train.loc[train["label"] == "true", "label"] = "truth"
train.loc[train["label"] == "false", "label"] = "fake"

dict_train = train.to_dict()
title_text = []
label = []
original_title = list(dict_train['title'].values())
original_text = list(dict_train['text'].values())
original_label = list(dict_train['label'].values())

for i in range(len(original_label)):
    if type(original_title[i]) != str or type(original_text[i]) != str:
        continue
    tt = original_title[i] + ' ' + original_text[i]
    title_text.append(tt)
    label.append(original_label[i])

final_train = {'title_text': title_text, 'label': label}
final_train = pd.DataFrame(data=final_train)
final_train = final_train.sample(frac = 1)

true_count = len(final_train[final_train['label']=='truth'])
false_count = len(final_train[final_train['label']=='fake'])
partial_count = len(final_train[final_train['label']=='partial'])
other_count = len(final_train[final_train['label']=='other'])

print("original training dataset statistics:")
print("# of samples that constitutes a demonstrably true statement: ", true_count)
print("# of samples that constitutes a fake statement: ", false_count)
print("# of samples that constitutes a statement that is mixture of truth and fake: ", partial_count)
print("# of samples that constitutes a statement that cannot be grouped as true or false: ", other_count)


preprocessed_folder = os.path.join(data_folder, "preprocessed")
try:
    os.makedirs(preprocessed_folder)
except FileExistsError:
    pass


preprocess_and_write(final_train, os.path.join(preprocessed_folder,"train.csv"))
