# News Recommender
#notlive

This recommender provides a bag of words from read articles(let's assume most liked articles)
these bag of words are then used to give recommendations

This recommender uses newspaper library to import newspaper articles from the website and then process it through the NLTK (Natural language processing toolkit) in the following steps ; 

Firstly, we have to install the newspaper3k module in our virtual environment, this can be done using ( pip install newspaper3k)
then simply import the newspaper and Article from newspaper

After that, we have to copy the url from a newspaper article and save it as a variable in our code, use Article method to help it read it as article.
After the above procedure is done we have to download and parse the article. using download() and parse() respectively, Thereafter vectorize the article using nlp()

now remember we have to tokenize and process the text and for that we'll have to import tokenizers and stemmers from sklearn library

I have defined functions for downloading, parsing the article and also to process the text. 
I made a function to dataframe the articles and tokens generated then merged all the dataframes together (these are however optional but it is convinient this way) and for this project it is atleast necessary to define a function for processing text (we'll later find out why)

from sklean library we have to import countvetorizer and tfidfTransformer ; it's function is to find the frequency of unique(processed words) in a single and across the document. but before that make sure to set the analyzer of countvectorize method as the function which you have defined to process the text.

by making a dataframe from the results we can find out the most frequent words across the documents (by sorting the datafieldl)

Now we can use normalize and nmf methods to find the cluster of words with most frequency and make a dataframe out of it. 

these steps will give the words that are most common cluster of words on articles read by you. Next step is to find articles having similar words.

For that we'll import our article database ( I have used 'historic_aricles of India database available on kaggle)
then we'll import Ktrain library fit the database in the model. then we'll train the recommender. 
you can input words to look for in a variable and have the model predict the related articles, but what I have done is left the input in hands of user to give. and the model will give related articles.
