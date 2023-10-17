import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.util import ngrams
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.stem import WordNetLemmatizer
import string
import warnings
warnings.filterwarnings('ignore')
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer, BertModel

import pickle
file=open("df_data.txt","rb")
df=pickle.load(file)

# Data Preprocessing
stop_words = set(stopwords.words('english'))
stemmer=PorterStemmer()
lemmatizer= WordNetLemmatizer()


def preprocess_text(text,remove_punctuation=True,remove_stopwords=True,stemming=False, lemmatization=False):
    if not isinstance(text,str):
        return text
    
    # For  removing punctuation 
    if remove_punctuation:
        text=text.translate(str.maketrans('','',string.punctuation))
    
    words=nltk.word_tokenize(text)
    
    # For Removing stopwords
    if remove_stopwords:
        words=[word for word in words if word.lower() not in stop_words]
        
    if stemming:
        words=[stemmer.stem(word) for word in words]
    elif lemmatization:
        words=[lemmatizer.lemmatize(word) for word in words]
        
    return ' '.join(words)

tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')

model=BertModel.from_pretrained('bert-base-uncased')
model.eval()

# Tokenizing and converting to BERT Embedings
def get_bert_embedding(text):
    tokens= tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs=model(**tokens)
    embeddings=outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings

# Function for calculating cosine similarity
from scipy.spatial.distance import cosine

def calculate_cosine_similarity(embedding1,embedding2):
    return 1 - cosine(embedding1,embedding2)
df['cosine_similarity']=df.apply(lambda row: calculate_cosine_similarity(row['text1_bert_embeddings'],row['text2_bert_embeddings']),axis=1)




# Function to convert cosine similarity to a range of 0 to 1
def Semantic_similarity_range(cosine_similarity_score):
    similarity_score=0.5*(cosine_similarity_score+1)
    return similarity_score
df['semantic_similarity']=df['cosine_similarity'].apply(Semantic_similarity_range)
