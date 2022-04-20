import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import newspaper
from newspaper import Article
import string
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

stopwords = nltk.corpus.stopwords.words('english')
ps = PorterStemmer()

def article_func(article):
    article.download()
    article.parse()
    article.nlp()

def process_text(article):    
    no_punct=''
    for word in article.text:
        if(word not in string.punctuation):
            no_punct = no_punct + word
    
    token = TreebankWordTokenizer().tokenize(no_punct)

    
    nostop = [word for word in token if word not in stopwords]

    new_items = [item for item in nostop if not item.isdigit()]
    
    text = [ps.stem(word) for word in new_items]
    
    return text

def dataframing(article):
    data = {'News':[article.text],'Processed Tokens':[process_text(article)]}
    df = pd.DataFrame(data)
    return df


url1 = 'https://www.deccanherald.com/opinion/the-economic-chaos-theory-of-riots-1101683.html'
article1 = Article(url1)
url2 = 'https://www.deccanherald.com/national/sc-scraps-bail-to-lakhimpur-kheri-accused-ashish-mishra-1101673.html'
article2 = Article(url2)
url3 = 'https://www.deccanherald.com/science-and-environment/what-s-the-new-omicron-xe-variant-and-should-you-be-worried-1101668.html'
article3 = Article(url3)
url4 = 'https://www.deccanherald.com/state/mangaluru/5-dead-after-inhaling-toxic-gas-at-mangaluru-fish-plant-1101651.html'
article4 = Article(url4)
url5 = 'https://www.deccanherald.com/international/world-news-politics/ukrainians-defy-russias-deadline-to-surrender-in-mariupol-1101567.html'
article5 = Article(url5)
url6 = 'https://www.deccanherald.com/international/russia-ukraine-crisis-live-war-putin-kyiv-maruipol-kherson-kharkiv-news-belarus-zelenskyy-lavrov-india-china-death-nuclear-1101429.html'
article6 = Article(url6)
url7 = 'https://www.deccanherald.com/international/zelenskyy-imf-managing-director-discuss-ukraines-post-war-reconstruction-1101641.html'
article7 = Article(url7)
url8='https://www.deccanherald.com/international/world-news-politics/ukraines-zelenskyy-condemns-shelling-as-bodies-line-streets-of-mariupol-1101658.html'
article8 = Article(url8)
url9 = 'https://www.deccanherald.com/city/top-bengaluru-stories/biggest-encroacher-of-bengaluru-lake-is-1101616.html'
article9 = Article(url9)

articles = (article1,article2,article3,article4,article5,article6,article7,article8,article9)
for article in articles:
    article_func(article)

df1 = dataframing(article1)
df2 = dataframing(article2)
df3 = dataframing(article3)
df4 = dataframing(article4)
df5 = dataframing(article5)
df6 = dataframing(article6)
df7 = dataframing(article7)
df8 = dataframing(article8)
df9 = dataframing(article9)

df = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9])

def process_text(txt):    
    no_punct=''
    for word in txt:
        if(word not in string.punctuation):
            no_punct = no_punct + word
    
    token = TreebankWordTokenizer().tokenize(no_punct)
    
    nostop = [word for word in token if word not in stopwords]
    
    
    new_items = [item for item in nostop if not item.isdigit()]
    
    text = [ps.stem(word) for word in new_items]
    
    return text

cv = CountVectorizer(analyzer= process_text)
x = cv.fit_transform(df['News'])


df_mat_count = pd.DataFrame(x.toarray(), columns=cv.get_feature_names())
print('Frequency of Words (Via Countvectorization) : ' ,df_mat_count.head(20))

from sklearn.feature_extraction.text import TfidfTransformer
vectorizer = TfidfTransformer()
vectorizer.fit(x)
freq = vectorizer.transform(x)
df_mat_freq = pd.DataFrame(freq.toarray(), columns=cv.get_feature_names())
print('Frequency of words across the whole document : ',df_mat_freq.head(20))

idf_df = pd.DataFrame(vectorizer.idf_,index = cv.get_feature_names(),columns=['idf_weights'])
idf_df.sort_values(by=['idf_weights']).head(25)

print('weightage of word across documnet from most to least : ',idf_df.head(25))


from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

# (optional) Disable FutureWarning of Scikit-learn
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# select number of topic clusters
n_topics = 10

# Create an NMF instance
nmf = NMF(n_components=n_topics)

# Fit the model to the tf_idf
nmf_features = nmf.fit_transform(freq)


# Create clustered dataframe the NMF clustered df
components = pd.DataFrame(
    nmf.components_, 
    columns=[df_mat_freq.columns]
    ) 

clusters = {}

# Show top 50 queries for each cluster
for i in range(len(components)):
    clusters[i] = []
    loop = dict(components.loc[i,:].nlargest(10)).items()
    for k,v in loop:
        clusters[i].append({'q':k[0],'sim_score': v})

# Create dataframe using the clustered dictionary
grouping = pd.DataFrame(clusters).T
grouping['topic'] = grouping[0].apply(lambda x: x['q'])
grouping.drop(0, axis=1, inplace=True)
grouping.set_index('topic', inplace=True)

print('Grouping', grouping)

def show_queries(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x['q'])
    return df

# Only display the query in the dataframe
clustered_queries = show_queries(grouping)
print('Clustered queries',clustered_queries)

text = pd.read_csv('historic_articles.csv')
text.dropna(axis=0, inplace=True)

texts = text['content']


import ktrain

tm = ktrain.text.get_topic_model(texts, n_features=10000)


tm.build(texts, threshold=0.90)

tm.train_recommender()
print('Give your input for recommendation : ')
rawtext = str(input())

tm.recommend(text=rawtext, n=5)

for i, doc in enumerate(tm.recommend(text = rawtext,n=5)):
    print('Result #%s'% (i+1))
    print('Text \n')
    print(' '.join(doc['text'].split()[:100]))

print('number of articles' ,len(texts))

t