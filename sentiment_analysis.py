#!/usr/bin/env python
# coding: utf-8

# In[205]:


# importing modules
import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from spacytextblob.spacytextblob import SpacyTextBlob


# In[206]:


#Load Spacy model
nlp = spacy.load('en_core_web_sm')


# In[207]:


#Add extension
nlp.add_pipe('spacytextblob')


# In[208]:


# Read amazon product reviews from the csv file
df = pd.read_csv('amazon_product_reviews.csv')


# In[209]:


#Only select the target column
df1 = df[['reviews.text']].copy()


# In[210]:


# Display the DataFrame and null values before processing
print("First 5 reviews:")
print(df1.head())
print("\nNumber of missing values in 'reviews.text':", df1['reviews.text'].isnull().sum())


# In[211]:


# Drop rows where 'reviews.text' is NaN
df1.dropna(subset=['reviews.text'], inplace=True)


# In[212]:


# Reset the index to ensure consistent indexing
df1.reset_index(drop=True, inplace=True)


# In[213]:


# Create function to strip out white spaces, punctuation, make everything lowercase & remove stop words
def clean_text(text):
  text = str(text).lower().strip()

  # Process the text with spaCy
  doc = nlp(text)

  cleaned_tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
  cleaned_text = ' '.join(cleaned_tokens) # To make the list of strings in cleaned_tokens one complete sentence again

  return cleaned_text


# In[214]:


# Create new column with preprocessed comments
df1['Cleaned_Reviews'] = df1['reviews.text'].apply(clean_text)
df1.head()


# In[215]:


# Grab one comment to check how polarity works
example = df1['Cleaned_Reviews'][9]
example


# In[216]:


doc = nlp(example)
polarity = doc._.blob.polarity
polarity


# In[217]:


# Function that uses polarity to deduce sentiment behind comment
def analyse_sentiment(text):
  doc = nlp(text)

  polarity = doc._.blob.polarity

  if polarity > 0:
    sentiment = 'Positive'
  elif polarity < 0:
    sentiment = 'Negative'
  else:
    sentiment = 'Neutral'

  return sentiment


# In[218]:


#Test the sentiment of a few reviews
comment3 = df1['Cleaned_Reviews'][0]
print(comment3)
analyse_sentiment(comment3)


# In[219]:


comment4 = df1['Cleaned_Reviews'][10]
print(comment4)
analyse_sentiment(comment4)


# In[220]:


comment5 = df1['Cleaned_Reviews'][20]
print(comment5)
analyse_sentiment(comment5)


# In[221]:


comment6 = df1['Cleaned_Reviews'][900]
print(comment6)
analyse_sentiment(comment6)


# In[222]:


# Grabbing 2 comments to use for sentiment analysis
comment1 = df1['Cleaned_Reviews'][17]
comment2 = df1['Cleaned_Reviews'][35]

comment1, comment2


# In[223]:


analyse_sentiment(comment1)


# In[224]:


analyse_sentiment(comment2)


# In[225]:


# Extra step: comparing similarity
def compare_similarity(comment1, comment2):
  doc1 = nlp(comment1)
  doc2 = nlp(comment2)
  similarity = doc1.similarity(doc2)

  return similarity


# In[226]:


compare_similarity(comment1, comment2)

