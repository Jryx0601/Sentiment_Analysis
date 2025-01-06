import numpy as np
import pandas as pd
import string
import nltk
stopwords = nltk.corpus.stopwords.words('english')
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
data = pd.read_excel("LabeledText.xlsx")
data_sentiment = data[['sentiment','Text']]

#Preprocessing dataset for model with stopwords
def remove_stopwords(txt_tokenized):
    txt_clean = [word for word in txt_tokenized if word not in stopwords]
    return txt_clean

#Exclamation Mark Removal
def remove_punctuation(txt):
    txt_nonpunct = [c for c in txt if c not in string.punctuation]
    return txt_nonpunct

import re
#Tokenization
def tokenize(txt):
    tokens = re.split('\W+', txt)
    return tokens

#Cleaning The Data
data_sentiment['Tokenize_text'] = data_sentiment['Text'].apply(lambda x: tokenize(x.lower()))
data_sentiment['Tokenize_text_Remove_Punctuation'] = data_sentiment['Tokenize_text'].apply(lambda x: remove_punctuation(x))
data_sentiment['Tokenize_text_Remove_Punctuation_Stopwords'] = data_sentiment['Tokenize_text_Remove_Punctuation'].apply(lambda x: remove_stopwords(x))

#Combination Function
def combine_txt(txt):
    txt_combine = ' '.join(txt)
    return txt_combine

data_sentiment['Final_Txt'] = data_sentiment['Tokenize_text_Remove_Punctuation_Stopwords'].apply(lambda x: combine_txt(x))
data_Final_Sentiment = data_sentiment[['sentiment','Final_Txt']]

def function_preprocessing(txt):
    tokenize_txt = tokenize(txt)
    stopwords_txt = remove_stopwords(tokenize_txt)
    final_txt = remove_punctuation(stopwords_txt)

    #Combine the following:
    Combined_txt = combine_txt(final_txt)
    return Combined_txt

#SVM
vectorizer = TfidfVectorizer()

x = vectorizer.fit_transform(data_Final_Sentiment['Final_Txt'])
y = data_Final_Sentiment['sentiment']

X_train,X_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.3)

svm = SVC(kernel='linear',C=1)
svm.fit(X_train,y_train)


def Prediction(txt):
    #Preprocess
    preprocess_txt = [function_preprocessing(txt['text'])]
    vectorizer_txt = vectorizer.transform(preprocess_txt)
    prediction = svm.predict(vectorizer_txt)
    prediction_final = prediction.tolist()
    return prediction_final