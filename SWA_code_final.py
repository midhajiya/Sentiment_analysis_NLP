#!/usr/bin/env python
# coding: utf-8

# In[1]:


# numerical computation
import numpy as np

# data processing/manipulation
import pandas as pd
pd.options.mode.chained_assignment = None
import re

# data visualization
import matplotlib.pyplot as plt
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import plotly.express as px


# In[2]:


import nltk  
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist

# spell correction, lemmatization
from textblob import TextBlob
from textblob import Word

# sklearn
from sklearn.model_selection import train_test_split


# In[3]:


# Loading each dataset
biden_df = pd.read_csv('hashtag_joebiden.csv', lineterminator='\n')


# In[4]:


#first 5 rows of biden_df
biden_df.head()


# In[5]:


biden_df.shape


# In[6]:


biden_df.describe()


# In[7]:


biden_df.info()


# In[10]:


# Remove unneeded columns
biden_df = biden_df.drop(columns=['tweet_id','user_id','user_name','user_screen_name','user_description','user_join_date','collected_at'])

# Renaming columns
biden_df = biden_df.rename(columns={"likes": "Likes", "retweet_count": "Retweets", "state": "State", "user_followers_count": "Followers"})

# Update United States country name for consistency
d = {"United States of America":"United States"}
biden_df['country'].replace(d, inplace=True)
biden_df = biden_df.loc[biden_df['country'] == "United States"]

# Drop null rows
biden_df = biden_df.dropna()


# In[11]:


#Preprocessing Tweets
to_remove = r'\d+|http?\S+|[^A-Za-z0-9]+'
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Function to preprocess tweet 
def clean_tweet(tweet, stem=False, lemmatize=False):

    # Make all text lowercase
    tweet = tweet.lower()
    
    # Remove links, special characters, punctuation, numbers, etc.
    tweet = re.sub(to_remove, ' ', tweet)
        
    filtered_tweet = []
    words = word_tokenize(tweet) 

    # Remove stopwords and stem
    for word in words:
        if not word in stop_words:
            if stem:
                filtered_tweet.append(ps.stem(word))
            elif lemmatize:
                filtered_tweet.append(Word(word).lemmatize())
            else:
                filtered_tweet.append(word)
            
    return filtered_tweet


# In[12]:


# Filtering all trump and biden tweets by applying cleantweet()
biden_df.tweet = biden_df.tweet.apply(lambda x: clean_tweet(x))


# In[13]:


#5 biden tweets after filtering
biden_df.tweet.head()


# In[14]:


# Sentiment Analysis
def sentiment_analysis(df):
    
    # Determine polarity and subjectivity
    df['Polarity'] = df['tweet'].apply(lambda x: TextBlob(' '.join(x)).sentiment.polarity)
    df['Subjectivity'] = df['tweet'].apply(lambda x: TextBlob(' '.join(x)).sentiment.subjectivity)
    
    # Classify overall sentiment
    df.loc[df.Polarity > 0,'Sentiment'] = 'positive'
    df.loc[df.Polarity == 0,'Sentiment'] = 'neutral'
    df.loc[df.Polarity < 0,'Sentiment'] = 'negative'
    
    return df[['tweet','Polarity','Subjectivity','Sentiment']].head()


# In[15]:


sentiment_analysis(biden_df)


# In[16]:


# Number of Tweets by Sentiment
# Overall sentiment breakdown - Biden 
print("Biden Tweet Sentiment Breakdown")

biden_positive = len(biden_df.loc[biden_df.Sentiment=='positive'])
biden_neutral = len(biden_df.loc[biden_df.Sentiment=='neutral'])
biden_negative = len(biden_df.loc[biden_df.Sentiment=='negative'])

print("Number of Positive Tweets: ", biden_positive)
print("Number of Neutral Tweets: ", biden_neutral)
print("Number of Negative Tweets: ", biden_negative)


# In[17]:


# Graphing the number of biden tweets by sentiment
data_b = {'Positive':biden_positive,'Neutral':biden_neutral,'Negative':biden_negative}
sentiment_b = list(data_b.keys()) 
num_tweets_b = list(data_b.values()) 

plt.figure(figsize = (8, 5)) 

plt.bar(sentiment_b, num_tweets_b, color ='blue', width = 0.5, edgecolor='black') 

plt.xlabel("Sentiment", fontweight ='bold') 
plt.ylabel("Number of Tweets", fontweight ='bold') 
plt.title("Biden Tweets by Sentiment", fontweight ='bold') 
plt.show() 


# In[18]:


# Calculate relative percentages by sentiment - Biden
total_tweets_b = len(biden_df.Sentiment)
prop_tweets_b = list(map(lambda x: round(x/total_tweets_b,2), num_tweets_b))


# In[19]:


# Graphing relative percentages of biden tweets
bar_width = 0.25
plt.subplots(figsize=(8,8))

br1 = np.arange(3) 
br2 = [x + bar_width for x in br1] 

b = plt.bar(br2, prop_tweets_b, color ='b', width = bar_width, edgecolor ='black', label ='Biden') 
   
plt.xlabel('Sentiment',fontweight ='bold') 
plt.ylabel('Percentage of Tweets',fontweight ='bold') 
plt.xticks([r + bar_width/2 for r in range(3)],['Positive','Neutral','Negative'])
plt.legend([b],['Percentage of Biden Tweets'])
plt.ylim(0.0, 1.0)
plt.title('Proportions of Tweets By Sentiment',fontweight ='bold')

plt.show()


# In[20]:


# Word Frequencies
# Function to return a string of all words in all tweets
def get_all_tweets(df,by_sentiment=False,sentiment="positive"):
    
    # Combine all words in tweets into a string
    if by_sentiment:
        if sentiment == "positive":
            words = ' '.join((df.loc[df.Sentiment=='positive'])['tweet'].apply(lambda x: ' '.join(x)))
        elif sentiment == "neutral":
            words = ' '.join((df.loc[df.Sentiment=='neutral'])['tweet'].apply(lambda x: ' '.join(x)))
        else:
            words = ' '.join((df.loc[df.Sentiment=='negative'])['tweet'].apply(lambda x: ' '.join(x)))
    else:
        words = ' '.join(df['tweet'].apply(lambda x: ' '.join(x)))
        
    return words


# In[21]:


# Create word strings
words_biden = get_all_tweets(biden_df)
words_pos_biden = get_all_tweets(biden_df,True,"positive")
words_neu_biden = get_all_tweets(biden_df,True,"neutral")
words_neg_biden = get_all_tweets(biden_df,True,"negative")

# Tokenize word strings
tokens_biden = word_tokenize(words_biden)
tokens_pos_biden = word_tokenize(words_pos_biden)
tokens_neu_biden = word_tokenize(words_neu_biden)
tokens_neg_biden = word_tokenize(words_neg_biden)


# In[22]:


# Function to plot most frequent words
def plot_word_freq(tokens,sentiment,t_or_b,color):
    fdist = FreqDist(tokens)
    fdist_df = pd.DataFrame(fdist.most_common(10), columns = ["Word","Frequency"])
    fig = px.bar(fdist_df, x="Word", y="Frequency",title="<b>Most Frequently Used Words in </b>" + sentiment + " " + t_or_b + "<b>-Related Tweets</b>")
    fig.update_traces(marker=dict(color=color),selector=dict(type="bar"),marker_line_color='black', marker_line_width=1.5, opacity=0.6)
    fig.show()


# In[23]:


# Most frequent words in all biden tweets
plot_word_freq(tokens_biden,"<b>ALL</b>","<b>Biden</b>","blue")


# In[24]:


# Most frequent words in positive biden tweets
plot_word_freq(tokens_pos_biden,"<b>POSITIVE</b>","<b>Biden</b>","green")


# In[25]:


# Most frequent words in neutral biden tweets
plot_word_freq(tokens_neu_biden,"<b>NEUTRAL</b>","<b>Biden</b>","blue")


# In[26]:


# Most frequent words in negative biden tweets
plot_word_freq(tokens_neg_biden,"<b>NEGATIVE</b>","<b>Biden</b>","red")


# In[29]:


# Function to generate word cloud
def create_wordcloud(words):
    
    # create wordcloud
    wordcloud = WordCloud(max_font_size=200, max_words=200, 
                          background_color="black").generate(words)

    # display the generated image
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# In[30]:


# Generate word cloud of biden tweets
create_wordcloud(words_biden)


# In[31]:


# Ploting polarity by state - Biden
fig = px.scatter(biden_df, x="State", y="Polarity", color="Polarity",title="<b>Biden-Related Tweet Polarity by State</b>",color_continuous_scale=px.colors.sequential.Inferno,width=1000, height=800)
fig.update_xaxes(categoryorder='category ascending')
fig.show()


# In[32]:


# Average polarity by state - Biden
biden_state_polarity = biden_df.groupby("State",as_index=False).mean()

fig = px.bar(biden_state_polarity, x="State", y="Polarity",
            title="<b>Average Polarity of Biden-Related Tweets by State</b>")
fig.update_traces(marker=dict(color="blue"),selector=dict(type="bar"),
                  marker_line_color='black', marker_line_width=0.8, opacity=0.6)
fig.show()


# In[33]:


# Polarity by Likes - Biden
fig = px.scatter(biden_df, x="Likes", y="Polarity", color="Polarity",title="<b>Biden-Related Tweet Polarity by Number of Likes</b>",color_continuous_scale=px.colors.sequential.Inferno,width=1000, height=800)
fig.show()


# In[34]:


# Polarity by Retweets - Biden
fig = px.scatter(biden_df, x="Retweets", y="Polarity", color="Polarity",title="<b>Biden-Related Tweet Polarity by Number of Retweets</b>",color_continuous_scale=px.colors.sequential.Inferno,width=1000, height=800)
fig.show()


# In[ ]:




