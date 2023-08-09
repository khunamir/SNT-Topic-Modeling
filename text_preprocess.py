import pandas as pd
import contractions
import re
import unicodedata
import inflect
import gensim
import sys
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from textblob import TextBlob
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS

def replace_contractions(text):
    ''' Replace contractions from each text in a pandas series '''
    print("Replacing contractions...")
    return contractions.fix(text)

def remove_url(text):
    ''' Remove URLs from each text in a pandas series '''
    print("Removing URLs...")
    return re.sub(r"http\S+", "", text)

def tokenize(data):
    ''' Tokenize texts into each word '''
    print("Tokenizing words...")
    tokenizer = TreebankWordTokenizer()
    token_list = []

    for text in data:
        tokens = tokenizer.tokenize(text)
        token_list.append(tokens)

    return token_list

def to_lower(tokens_list):
    ''' Lowercase all tokens ''' 
    print("Lowercasing tokens...")
    lowered_tokens_list = []
    
    for tokens in tokens_list:
        lowered_tokens = []

        for token in tokens:
            lowered_tokens.append(token.lower())

        lowered_tokens_list.append(lowered_tokens)

    return lowered_tokens_list

def remove_nonascii(tokens_list):
    ''' Remove non-ascii tokens '''
    print("Removing non-ascii ...")
    non_ascii_list = []

    for tokens in tokens_list:
        non_ascii_tokens = []
        
        for token in tokens:
            new_token = unicodedata.normalize('NFKD', token).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            non_ascii_tokens.append(new_token)

        non_ascii_list.append(non_ascii_tokens)

    return non_ascii_list

def remove_punctuations(tokens_list):
    ''' Remove punctuation tokens '''
    print("Removing punctuations...")
    rem_punctuations_list = []

    for tokens in tokens_list:
        rem_punctuations = []

        for token in tokens:
            new_token = re.sub(r'[^\w\s]', '', token)
            
            if new_token != '':
                rem_punctuations.append(new_token)

        rem_punctuations_list.append(rem_punctuations)

    return rem_punctuations_list

def replace_number(tokens_list):
    ''' Replace number tokens into character tokens '''
    ''' eg. 100 -> one hundred'''
    print("Replacing numbers...")
    p = inflect.engine()
    replaced_number_list = []

    for tokens in tokens_list:
        replaced_number = []

        for token in tokens:
            if token.isdigit():
                try:
                    new_token = p.number_to_words(token)
                    replaced_number.append(new_token)
                except Exception:
                    pass    # ignore extra huge numbers
            else:
                replaced_number.append(token)

        replaced_number_list.append(replaced_number)

    return replaced_number_list

def stopwords_removal(tokens_list):
    ''' Remove stop words tokens '''
    print("Removing stop words...")
    #stop_words = set(stopwords.words('english'))
    filtered_tokens = []

    for tokens in tokens_list:
        #filtered_tokens.append([word for word in tokens if word not in stop_words])
        sentence = ' '.join(tokens)
        new_sentence = remove_stopwords(sentence)
        filtered_tokens.append(new_sentence.split())

    return filtered_tokens

def lemmatize(tokens_list):
    ''' Reduce tokens into lemmas based on POS tags '''
    print("Reducing tokens into lemmas...")
    lemmas_list = []

    for tokens in tokens_list:
        sentence = ' '.join(tokens)
        words_tags = pos_tagger(sentence)
        lemmas = [w.lemmatize(tag) for w, tag in words_tags]
        lemmas_list.append(lemmas)

    return lemmas_list

def pos_tagger(sentence):
    ''' Tag each word in a sentence with a POS tag for lemmatizing '''
    sent = TextBlob(sentence)
    tag_dict = {"J": 'a', "N": 'n', "V": 'v', "R": 'r'}
    words_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]

    return words_tags

def normalize(tokens_list):
    ''' Normalize tokens, receive parameter in a form of list of tokens from each text '''    
    tokens_list = remove_nonascii(tokens_list)
    tokens_list = to_lower(tokens_list)
    tokens_list = remove_punctuations(tokens_list) # REPORT: Kena remove punctuation dulu sebab akan ada nombor dengan koma kat tengah eg :- 10,000
    tokens_list = replace_number(tokens_list)
    tokens_list = lemmatize(tokens_list)
    tokens_list = stopwords_removal(tokens_list) # REPORT: NLTK punya stopwords tak ckup
                                                 # REPORT: Kena lemmatize dulu baru stopword, sebab some stop words in lemmas

    return tokens_list

def preprocess(text):
    ''' Preprocess data, receive parameter in a form of pandas Series'''
    print("Removing contractions...")
    text.apply(lambda x: contractions.fix(x))
    print("Removing URLs...")
    text.apply(lambda x: re.sub(r"http\S+", "", x))
    
    tokens = tokenize(text)

    return normalize(tokens)

if __name__ == "__main__":

    path = sys.argv[1]
    new_path = path.replace('.pkl','')
    data = pd.read_pickle(path)
    #data = pd.read_pickle('./PopSciData/SciTechData.pickle')
    processed = preprocess(data.text)
    data.to_pickle(f'{new_path}_processed.pkl')
