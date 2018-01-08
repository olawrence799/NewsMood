
# Observed Trends
    1) CBS has by far the most positive 'compound sentiment' over the last 100 tweets according to Vader sentiment analysis.
    2) BBC and NY Times are the only two out of the five observed news outlets that have negative 'compound sentiment' values over the last 100 tweets. BBC is the most negative overall -- perhaps because they have the most reporting on world news out of these observed news outlets.
    3) Fox News has the aggregate compound score closest to zero, implying that their tweets are the most neutral overall in this dataset.


```python
import tweepy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

consumer_key = "RZtUlr4WgwHqb25tlJzQFlgwH"
consumer_secret = "WyHARdwkWASvQyyxWZlLU6fRmCWnUuBqQn9Ngkg8TgikAQvB9u"
access_token = "942946269793624065-0ODdw1IsP0ZzDr3hMYi1NaUjJNGIQIt"
access_token_secret = "TRsMADWtzgd3QAbyzYmmJBCiQcRmIVzMdxTVBQuj2Zh9h"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

target_users = ['@BBCWorld', '@CBS', '@CNN', '@FoxNews', '@nytimes']
all_user_list = []
all_timestamp_list = []
all_compound_list = []
all_pos_list = []
all_neu_list = []
all_neg_list = []
all_text_list = []
all_sentiment_list = []
all_tweets_ago_list = []

for target_user in target_users:
    user_list = []
    timestamp_list = []
    compound_list = []
    pos_list = []
    neu_list = []
    neg_list = []
    text_list = []
    tweets_ago_list = []
    tweet_count = 0
    
    for x in range(5):
        public_tweets = api.user_timeline(target_user, page=x)

        for tweet in public_tweets:
            timestamp = tweet["created_at"]
            compound = analyzer.polarity_scores(tweet["text"])["compound"]
            pos = analyzer.polarity_scores(tweet["text"])["pos"]
            neu = analyzer.polarity_scores(tweet["text"])["neu"]
            neg = analyzer.polarity_scores(tweet["text"])["neg"]
            user_list.append(tweet["user"]["screen_name"])
            timestamp_list.append(timestamp)
            compound_list.append(compound)
            pos_list.append(pos)
            neu_list.append(neu)
            neg_list.append(neg)
            text_list.append(tweet["text"])
            tweet_count += 1
            tweets_ago_list.append(tweet_count)
            #print(tweet['user']['screen_name'])
            #print(json.dumps(tweet, sort_keys=True, indent=4, separators=(',', ': ')))
            
    all_user_list.append(user_list)
    all_timestamp_list.append(timestamp_list)
    all_compound_list.append(compound_list)
    all_pos_list.append(pos_list)
    all_neu_list.append(neu_list)
    all_neg_list.append(neg_list)
    all_text_list.append(text_list)
    all_tweets_ago_list.append(tweets_ago_list)
            
        # Print the Averages
            #print("")
            #print("User: %s" % target_term)
            #print(timestamp)
            #print(f"Compound: {np.mean(compound_list)}")
            
    #sentiment = {
    #    "%s Compound" % target_user : np.mean(compound_list)
    #}
    sentiments = np.mean(compound_list)
    all_sentiment_list.append(sentiments)
    #print (sentiment)
```


```python
all_user_list = np.array(all_user_list).flatten().tolist()
all_timestamp_list = np.array(all_timestamp_list).flatten().tolist()
all_compound_list = np.array(all_compound_list).flatten().tolist()
all_pos_list = np.array(all_pos_list).flatten().tolist()
all_neu_list = np.array(all_neu_list).flatten().tolist()
all_neg_list = np.array(all_neg_list).flatten().tolist()
all_text_list = np.array(all_text_list).flatten().tolist()
all_tweets_ago_list = np.array(all_tweets_ago_list).flatten().tolist()
```


```python
sentiment = {'User': all_user_list, 'Timestamp': all_timestamp_list, 'Compound_Score': all_compound_list, 'Pos_Score': all_pos_list, 'Neu_Score': all_neu_list, 'Neg_Score': all_neg_list, 'Tweets_Ago': all_tweets_ago_list, 'Tweet_Text': all_text_list}
sentiment_df = pd.DataFrame(sentiment)
sentiment_df.to_csv("Twitter_News_Outlet_Sentiment.csv", index=False, header=True)
sentiment_df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound_Score</th>
      <th>Neg_Score</th>
      <th>Neu_Score</th>
      <th>Pos_Score</th>
      <th>Timestamp</th>
      <th>Tweet_Text</th>
      <th>Tweets_Ago</th>
      <th>User</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.5574</td>
      <td>0.118</td>
      <td>0.882</td>
      <td>0.000</td>
      <td>Mon Jan 08 22:31:10 +0000 2018</td>
      <td>RT @BBCSport: Widnes Vikings' Kato Ottio has d...</td>
      <td>1</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Mon Jan 08 22:21:04 +0000 2018</td>
      <td>North and South Korea to begin high-level talk...</td>
      <td>2</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Mon Jan 08 22:07:47 +0000 2018</td>
      <td>Google sued over 'male discrimination' https:/...</td>
      <td>3</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Mon Jan 08 21:22:53 +0000 2018</td>
      <td>Rugby League player Ottio dies aged 23 https:/...</td>
      <td>4</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.3923</td>
      <td>0.164</td>
      <td>0.744</td>
      <td>0.092</td>
      <td>Mon Jan 08 21:20:13 +0000 2018</td>
      <td>RT @BBCNorthAmerica: - Sad!\n- Bigly? Or big l...</td>
      <td>5</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.7717</td>
      <td>0.242</td>
      <td>0.758</td>
      <td>0.000</td>
      <td>Mon Jan 08 21:09:40 +0000 2018</td>
      <td>RT @BBCSport: Widnes Vikings centre Kato Ottio...</td>
      <td>6</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.5719</td>
      <td>0.485</td>
      <td>0.515</td>
      <td>0.000</td>
      <td>Mon Jan 08 19:26:48 +0000 2018</td>
      <td>'Raw water': A dangerous new health craze? htt...</td>
      <td>7</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Mon Jan 08 18:44:08 +0000 2018</td>
      <td>Ghana bars recruits over stretch marks and ble...</td>
      <td>8</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.8176</td>
      <td>0.586</td>
      <td>0.414</td>
      <td>0.000</td>
      <td>Mon Jan 08 18:36:15 +0000 2018</td>
      <td>Egypt police detained after custody death trig...</td>
      <td>9</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.5574</td>
      <td>0.340</td>
      <td>0.660</td>
      <td>0.000</td>
      <td>Mon Jan 08 18:23:20 +0000 2018</td>
      <td>Bangladesh court upholds Myanmar Rohingya marr...</td>
      <td>10</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-0.5423</td>
      <td>0.304</td>
      <td>0.696</td>
      <td>0.000</td>
      <td>Mon Jan 08 18:15:21 +0000 2018</td>
      <td>Prisoner wakes up in mortuary in Asturias Spai...</td>
      <td>11</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Mon Jan 08 17:34:37 +0000 2018</td>
      <td>RT @awzurcher: Winfrey vs Trump may not be the...</td>
      <td>12</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>12</th>
      <td>-0.3182</td>
      <td>0.277</td>
      <td>0.723</td>
      <td>0.000</td>
      <td>Mon Jan 08 17:04:17 +0000 2018</td>
      <td>A child interrupts an Al Jazeera interview htt...</td>
      <td>13</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Mon Jan 08 16:51:06 +0000 2018</td>
      <td>Juncker: Don't believe Brexit won't happen htt...</td>
      <td>14</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>14</th>
      <td>-0.0516</td>
      <td>0.107</td>
      <td>0.893</td>
      <td>0.000</td>
      <td>Mon Jan 08 16:35:02 +0000 2018</td>
      <td>Trump to say 200,000 El Salvadorans must leave...</td>
      <td>15</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>15</th>
      <td>-0.6908</td>
      <td>0.439</td>
      <td>0.561</td>
      <td>0.000</td>
      <td>Mon Jan 08 16:03:09 +0000 2018</td>
      <td>Somaliland passes first law against rape https...</td>
      <td>16</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Mon Jan 08 15:47:31 +0000 2018</td>
      <td>Iran bans English from being taught in primary...</td>
      <td>17</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Mon Jan 08 15:00:59 +0000 2018</td>
      <td>Firefighters tackle Trump Tower blaze in New Y...</td>
      <td>18</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.6486</td>
      <td>0.000</td>
      <td>0.773</td>
      <td>0.227</td>
      <td>Mon Jan 08 14:49:50 +0000 2018</td>
      <td>Oprah for President? Her passionate speech abo...</td>
      <td>19</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Mon Jan 08 14:49:24 +0000 2018</td>
      <td>Apple investors urge action on 'smartphone add...</td>
      <td>20</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>20</th>
      <td>-0.5574</td>
      <td>0.118</td>
      <td>0.882</td>
      <td>0.000</td>
      <td>Mon Jan 08 22:31:10 +0000 2018</td>
      <td>RT @BBCSport: Widnes Vikings' Kato Ottio has d...</td>
      <td>21</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Mon Jan 08 22:21:04 +0000 2018</td>
      <td>North and South Korea to begin high-level talk...</td>
      <td>22</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Mon Jan 08 22:07:47 +0000 2018</td>
      <td>Google sued over 'male discrimination' https:/...</td>
      <td>23</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Mon Jan 08 21:22:53 +0000 2018</td>
      <td>Rugby League player Ottio dies aged 23 https:/...</td>
      <td>24</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>24</th>
      <td>-0.3923</td>
      <td>0.164</td>
      <td>0.744</td>
      <td>0.092</td>
      <td>Mon Jan 08 21:20:13 +0000 2018</td>
      <td>RT @BBCNorthAmerica: - Sad!\n- Bigly? Or big l...</td>
      <td>25</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>25</th>
      <td>-0.7717</td>
      <td>0.242</td>
      <td>0.758</td>
      <td>0.000</td>
      <td>Mon Jan 08 21:09:40 +0000 2018</td>
      <td>RT @BBCSport: Widnes Vikings centre Kato Ottio...</td>
      <td>26</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>26</th>
      <td>-0.5719</td>
      <td>0.485</td>
      <td>0.515</td>
      <td>0.000</td>
      <td>Mon Jan 08 19:26:48 +0000 2018</td>
      <td>'Raw water': A dangerous new health craze? htt...</td>
      <td>27</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Mon Jan 08 18:44:08 +0000 2018</td>
      <td>Ghana bars recruits over stretch marks and ble...</td>
      <td>28</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>28</th>
      <td>-0.8176</td>
      <td>0.586</td>
      <td>0.414</td>
      <td>0.000</td>
      <td>Mon Jan 08 18:36:15 +0000 2018</td>
      <td>Egypt police detained after custody death trig...</td>
      <td>29</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>29</th>
      <td>-0.5574</td>
      <td>0.340</td>
      <td>0.660</td>
      <td>0.000</td>
      <td>Mon Jan 08 18:23:20 +0000 2018</td>
      <td>Bangladesh court upholds Myanmar Rohingya marr...</td>
      <td>30</td>
      <td>BBCWorld</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>470</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Mon Jan 08 13:46:07 +0000 2018</td>
      <td>Natalie Portman delivered a line that instantl...</td>
      <td>71</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>471</th>
      <td>0.0258</td>
      <td>0.155</td>
      <td>0.687</td>
      <td>0.159</td>
      <td>Mon Jan 08 13:32:06 +0000 2018</td>
      <td>"The Globes still celebrated the power players...</td>
      <td>72</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>472</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Mon Jan 08 13:00:28 +0000 2018</td>
      <td>Morning Briefing: Here's what you need to know...</td>
      <td>73</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>473</th>
      <td>-0.7096</td>
      <td>0.296</td>
      <td>0.704</td>
      <td>0.000</td>
      <td>Mon Jan 08 12:45:15 +0000 2018</td>
      <td>Iran has banned the teaching of English in pri...</td>
      <td>74</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>474</th>
      <td>-0.7003</td>
      <td>0.254</td>
      <td>0.746</td>
      <td>0.000</td>
      <td>Mon Jan 08 12:31:05 +0000 2018</td>
      <td>A senior BBC News editor accused the network o...</td>
      <td>75</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>475</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Mon Jan 08 12:15:09 +0000 2018</td>
      <td>A transcript of Seth Meyers's Golden Globes mo...</td>
      <td>76</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>476</th>
      <td>0.2732</td>
      <td>0.000</td>
      <td>0.890</td>
      <td>0.110</td>
      <td>Mon Jan 08 12:00:21 +0000 2018</td>
      <td>President Trump defended his fitness for offic...</td>
      <td>77</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>477</th>
      <td>0.0258</td>
      <td>0.000</td>
      <td>0.942</td>
      <td>0.058</td>
      <td>Mon Jan 08 11:50:08 +0000 2018</td>
      <td>Stephen Bannon tried backing away from his exp...</td>
      <td>78</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>478</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Mon Jan 08 11:40:13 +0000 2018</td>
      <td>Read speeches from Elisabeth Moss, Laura Dern,...</td>
      <td>79</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>479</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Mon Jan 08 11:30:06 +0000 2018</td>
      <td>Morning Briefing: Here's what you need to know...</td>
      <td>80</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>480</th>
      <td>-0.2732</td>
      <td>0.091</td>
      <td>0.909</td>
      <td>0.000</td>
      <td>Mon Jan 08 11:21:02 +0000 2018</td>
      <td>An Iranian oil tanker that collided with anoth...</td>
      <td>81</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>481</th>
      <td>0.4215</td>
      <td>0.000</td>
      <td>0.877</td>
      <td>0.123</td>
      <td>Mon Jan 08 11:11:02 +0000 2018</td>
      <td>RT @nytimesworld: Remember the “new Coke” blun...</td>
      <td>82</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>482</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Mon Jan 08 11:00:23 +0000 2018</td>
      <td>Israel published a blacklist of 20 organizatio...</td>
      <td>83</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>483</th>
      <td>0.5267</td>
      <td>0.000</td>
      <td>0.784</td>
      <td>0.216</td>
      <td>Mon Jan 08 10:46:03 +0000 2018</td>
      <td>The strongest challenger to Egypt’s leader wit...</td>
      <td>84</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>484</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Mon Jan 08 10:29:02 +0000 2018</td>
      <td>Read Oprah Winfrey's Golden Globes speech http...</td>
      <td>85</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>485</th>
      <td>-0.4215</td>
      <td>0.128</td>
      <td>0.872</td>
      <td>0.000</td>
      <td>Mon Jan 08 10:15:12 +0000 2018</td>
      <td>The police in Belfast said they were investiga...</td>
      <td>86</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>486</th>
      <td>0.4767</td>
      <td>0.000</td>
      <td>0.763</td>
      <td>0.237</td>
      <td>Mon Jan 08 09:59:03 +0000 2018</td>
      <td>The complete list of winners at the 2018 Golde...</td>
      <td>87</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>487</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Mon Jan 08 09:44:04 +0000 2018</td>
      <td>RT @nytimesworld: How do you maintain a fence ...</td>
      <td>88</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>488</th>
      <td>-0.4404</td>
      <td>0.146</td>
      <td>0.854</td>
      <td>0.000</td>
      <td>Mon Jan 08 09:31:01 +0000 2018</td>
      <td>Opinion: "America is upside down and inside ou...</td>
      <td>89</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>489</th>
      <td>-0.2500</td>
      <td>0.100</td>
      <td>0.900</td>
      <td>0.000</td>
      <td>Mon Jan 08 09:16:05 +0000 2018</td>
      <td>Countries are unsure whether to take President...</td>
      <td>90</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>490</th>
      <td>-0.7096</td>
      <td>0.296</td>
      <td>0.704</td>
      <td>0.000</td>
      <td>Mon Jan 08 08:52:06 +0000 2018</td>
      <td>Iran has banned the teaching of English in pri...</td>
      <td>91</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>491</th>
      <td>0.6187</td>
      <td>0.082</td>
      <td>0.690</td>
      <td>0.229</td>
      <td>Mon Jan 08 08:37:05 +0000 2018</td>
      <td>RT @FrankBruni: In this anxious time of absent...</td>
      <td>92</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>492</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Mon Jan 08 08:22:05 +0000 2018</td>
      <td>Heavy snowfall. Face-freezing temperatures. Ar...</td>
      <td>93</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>493</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Mon Jan 08 08:07:11 +0000 2018</td>
      <td>Will Steve Bannon be able to hang on to Breitb...</td>
      <td>94</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>494</th>
      <td>0.4019</td>
      <td>0.000</td>
      <td>0.891</td>
      <td>0.109</td>
      <td>Mon Jan 08 07:52:01 +0000 2018</td>
      <td>RT @nytimesarts: The Golden Globes was a half-...</td>
      <td>95</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>495</th>
      <td>0.5106</td>
      <td>0.061</td>
      <td>0.711</td>
      <td>0.228</td>
      <td>Mon Jan 08 07:37:06 +0000 2018</td>
      <td>Can tech companies like Google and Paytm convi...</td>
      <td>96</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>496</th>
      <td>0.5719</td>
      <td>0.000</td>
      <td>0.709</td>
      <td>0.291</td>
      <td>Mon Jan 08 07:22:06 +0000 2018</td>
      <td>The votes are in: Oprah Winfrey won the night ...</td>
      <td>97</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>497</th>
      <td>0.5859</td>
      <td>0.000</td>
      <td>0.840</td>
      <td>0.160</td>
      <td>Mon Jan 08 07:07:03 +0000 2018</td>
      <td>RT @sewellchan: I was moved by @Oprah's remark...</td>
      <td>98</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>498</th>
      <td>0.3818</td>
      <td>0.000</td>
      <td>0.874</td>
      <td>0.126</td>
      <td>Mon Jan 08 06:52:00 +0000 2018</td>
      <td>Walking or cycling changes the makeup of our g...</td>
      <td>99</td>
      <td>nytimes</td>
    </tr>
    <tr>
      <th>499</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Mon Jan 08 06:37:04 +0000 2018</td>
      <td>A suburb in Australia just had the hottest day...</td>
      <td>100</td>
      <td>nytimes</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 8 columns</p>
</div>




```python
users = sentiment_df['User'].unique()
colors = ['yellow', 'lightskyblue', 'darkblue', 'red', 'green']

for i in range(len(users)):
    plt.scatter(x=sentiment_df[sentiment_df['User']==users[i]]['Tweets_Ago'].values,
                y=sentiment_df[sentiment_df['User']==users[i]]['Compound_Score'].values,
                s = 90,#*sentiment_df[sentiment_df['User']==users[i]]['Tweets_Ago'].values,
                c = colors[i], label = users[i],
                alpha = .7, edgecolor = 'black', linewidth = .8)

plt.xlabel("Tweets Ago")
plt.ylabel("Tweet Polarity")
plt.title("Sentiment Analysis of Media Tweets (01/08/17)")
plt.legend(title="Media Sources", loc='upper right')
plt.grid()
plt.gcf().set_size_inches(15, 6)
plt.rcParams['axes.facecolor'] = 'gainsboro'
ax = plt.gca()
ax.set_xlim(ax.get_xlim()[::-1])
plt.ylim(-1, 1)
plt.savefig("Sentiment_Analysis_Tweets.png")
plt.show()
```


![png](output_4_0.png)



```python
avg_sentiment = {'User': target_users, 'Avg_Compound_Score': all_sentiment_list}
avg_sentiment_df = pd.DataFrame(avg_sentiment)
avg_sentiment_df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Avg_Compound_Score</th>
      <th>User</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.051442</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.316011</td>
      <td>@CBS</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.088324</td>
      <td>@CNN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.010650</td>
      <td>@FoxNews</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.024022</td>
      <td>@nytimes</td>
    </tr>
  </tbody>
</table>
</div>




```python
x_values = np.arange(len(avg_sentiment_df))
plt.figure(figsize=(10, 7))
barlist=plt.bar(x_values, avg_sentiment_df["Avg_Compound_Score"], alpha=0.5, align='center', width=1)
plt.xticks(x_values, avg_sentiment_df["User"], rotation="horizontal")
for i in range(len(barlist)):
    barlist[i].set_color(colors[i])
plt.ylabel("Tweet Polarity")
plt.title("Overall Media Sentiment Based On Twitter (01/08/17)")
plt.savefig("Overall_Media_Sentiment.png")
plt.show()
```


![png](output_6_0.png)

