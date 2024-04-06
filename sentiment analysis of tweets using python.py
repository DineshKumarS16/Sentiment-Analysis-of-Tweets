#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install scikit-learn')


# In[2]:


import sklearn 
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import string
import nltk
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# In[3]:


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# In[4]:


data=pd.read_csv("train_tweet.csv")
data.head(31000)


# In[5]:


#datatype info
data.info(31000)


# In[6]:


# removes pattern in the input text
def remove_pattern(input_txt,pattern):
    r=re.findall(pattern,input_txt)
    for word in r:
        input_txt=re.sub(word,"",input_txt)
    return input_txt


# In[7]:


data.head(31000)


# In[8]:


# remove twitter handles (@user) 
data['clean_tweet']=np.vectorize(remove_pattern)(data['tweet'],"@[\w]*")


# In[9]:


data.head(10000)


# In[10]:


# remove special characters, numbers and punctuations 
data['clean_tweet'] = data['clean_tweet'].str.replace("[^a-zA-Z#]", " ")


# In[11]:


data.head(10000)


# In[12]:


# remove short words
data['clean_tweet']=data['clean_tweet'].apply(lambda x:" ".join([w for w in x.split() if len(w)>3]))


# In[13]:


data.head(10000)


# In[14]:


# individual words considered as tokens
tokenized_tweet=data['clean_tweet'].apply(lambda x: x.split())
tokenized_tweet.head(10000)


# In[15]:


# stem the words
tokenized_tweet=data['clean_tweet'].apply(lambda x: x.split())
tokenized_tweet.head(10000)


# In[16]:


# combine words into a single sentence
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i]=" ".join(tokenized_tweet[i])
data['clean_tweet']=tokenized_tweet
data.head(10000)


# In[17]:


get_ipython().system('pip install wordcloud')


# In[18]:


# visualize the frequent words
all_words= " ".join([sentence for sentence  in data['clean_tweet']])

from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

# plot the graph
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show(10000)


# In[19]:


# frequent words visualization for positive
all_words= " ".join([sentence for sentence  in data['clean_tweet'][data['label']==0]])


wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show(10000)


# In[20]:


# frequent words visualization for negative
all_words= " ".join([sentence for sentence  in data['clean_tweet'][data['label']==1]])


wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show(10000)


# In[21]:


# extract the hashtag
def hashtag_extract(tweets):
    hashtags=[]
    # loop words in the tweet
    for tweet in tweets:
        ht=re.findall(r"#(\w+)",tweet)
        hashtags.append(ht)
    return hashtags


# In[22]:


# extract hashtags from non-racist/sexist tweets
ht_positive=hashtag_extract(data['clean_tweet'][data['label']==0])

# extract hashtags from racist/sexist tweets
ht_negative=hashtag_extract(data['clean_tweet'][data['label']==1])


# In[23]:


ht_positive[:5]


# In[24]:


# unnest list
ht_positive=sum(ht_positive,[])
ht_negative=sum(ht_negative,[])


# In[25]:


ht_positive[:5]


# In[26]:


freq=nltk.FreqDist(ht_positive)
d=pd.DataFrame({'Hashtag':list(freq.keys()),
               'Count':list(freq.values())})
d.head(10000)


# In[27]:


# select top 10 hashtags
d=d.nlargest(columns='Count', n=10)
plt.figure(figsize=(15,9))
sns.barplot(data=d,x='Hashtag',y='Count')
plt.show(10000)


# In[28]:


freq=nltk.FreqDist(ht_negative)
d=pd.DataFrame({'Hashtag':list(freq.keys()),
               'Count':list(freq.values())})
d.head(10000)


# In[29]:


# select top 10 hashtags
d=d.nlargest(columns='Count', n=10)
plt.figure(figsize=(15,9))
sns.barplot(data=d,x='Hashtag',y='Count')
plt.show(10000)


# In[30]:


# feature extraction
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer= CountVectorizer()
bow = bow_vectorizer.fit_transform(data['clean_tweet'])


# In[31]:


bow[0].toarray()


# In[32]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(bow, data['label'], random_state=42,test_size=0.25)


# In[33]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score


# In[34]:


# training
model=LogisticRegression()
model.fit(x_train, y_train)


# In[35]:


# testing
pred = model.predict(x_test)
f1_score(y_test,pred)


# In[36]:


accuracy_score(y_test,pred)


# In[37]:


# use probability to get output
pred_prob = model.predict_proba(x_test)
pred=pred_prob[:, 1]>=0.3
pred=pred.astype(np.int)

f1_score(y_test,pred)


# In[38]:


accuracy_score(y_test,pred)


# In[39]:


pred_prob[0][1]>=0.3


# In[ ]:




