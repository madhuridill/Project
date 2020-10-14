# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 13:09:37 2020

@author: madhu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, naive_bayes, svm


df = pd.read_csv('CNN_news.csv')

df.head(10)
df.columns

#fetch word cloud for each text present in the data set
df['Wordcount'] = df['TEXT'].apply(lambda x: len(str(x).split(" ")))
df[['TEXT','Wordcount']].head()

#descriptive statistics for the word count
df.Wordcount.describe()
#plot news text length
df['News Text Length'] = df['TEXT'].str.len()
plt.figure(figsize=(12.8,6))
sns.distplot(df['News Text Length']).set_title('News Length Text Distribution');
df['News Text Length'].describe()
#find common words
Common_Words = pd.Series(' '.join(df['TEXT']).split()).value_counts()[:20]
Common_Words

#find uncommom words
Uncommon_words =  pd.Series(' '.join(df ['TEXT']).split()).value_counts()[-20:]
Uncommon_words

#remove possive pronoun

df['TEXT'] = df['TEXT'].str.replace("'s", "")

# Libraries for text preprocessing
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
#nltk.download('wordnet') 
from nltk.stem.wordnet import WordNetLemmatizer
#stop words
Stopwords = set(stopwords.words("english"))
#adding more words to stop words
##Creating a list of custom stopwords
addNewStopWords = ['said','say','also','could','even','get','would','like','one','caption','may','much','go','make','come','take'
                   ,'know','well','really','much','two','must','ago','new','many','say','way','told']
Stopwords = Stopwords.union(addNewStopWords)



corpus = []
for i in range(0, 3847):
    #Remove punctuations
    text = re.sub('[^a-zA-Z]', ' ', df['TEXT'][i])
    
    #Convert to lowercase
    text = text.lower()
    
    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    ##Convert to list from string
    text = text.split()
    
    ##Stemming
    ps=PorterStemmer()
    #Lemmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in  
            Stopwords] 
    text = " ".join(text)
    corpus.append(text)
    
#Word cloud
#!pip install wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=Stopwords,
                          max_words=100,
                          max_font_size=50, 
                          random_state=42
                         ).generate(str(corpus))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)

#check subjectivity and polarity of news text
from textblob import TextBlob



polarity=[]
subjectivity=[]
df['Polarity']=df.TEXT.apply(lambda x: TextBlob(x).sentiment.polarity)
df['Subj']=df.TEXT.apply(lambda x: TextBlob(x).sentiment.subjectivity)
df.head(10)
#Create 2 arrays
polarity=[]
subj=[]

#Get polarity and sentiment for each row and put it in either polarity or sentiment 
for t in df.TEXT:
    tx=TextBlob(t)
    polarity.append(tx.sentiment.polarity)
    subj.append(tx.sentiment.subjectivity)
#Put in dataframe polsubj which has a column of polarity values and a column of subjectivity values
polsubj = pd.DataFrame({'polarity': polarity,'subjectivity': subj})
#Plot the line graph
polsubj.plot(title='Polarity and Subjectivity')
#adding changes in text to original data
df['New Text'] = corpus


#feature extraction 
#word count vector for text preperation
from sklearn.feature_extraction.text import CountVectorizer
import re
cv=CountVectorizer(max_df=0.8,stop_words=Stopwords, max_features=10000, ngram_range=(1,3))
X=cv.fit_transform(corpus)
X.shape
list(cv.vocabulary_.keys())[:10]

#see the words with top ngram, bigram and trigram
#frequently occuring words that is common in the text column 
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    return words_freq[:n]
#put common words in pandas data frame
top_words = get_top_n_words(corpus, n=20)
top_df = pd.DataFrame(top_words)
top_df.columns=["Word", "Freq"]

#visulalize the bar plot using seaborn library - one grams
sns.set(rc={'figure.figsize':(13,8)})
g = sns.barplot(x="Word", y="Freq", data=top_df)
g.set_xticklabels(g.get_xticklabels(), rotation=30)

#visulalize the bar plot using seaborn library - bi grams
def get_top_n2_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2,2),  
            max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]
top2_words = get_top_n2_words(corpus, n=20)
top2_df = pd.DataFrame(top2_words)
top2_df.columns=["Bi-gram", "Freq"]
print(top2_df)
#Barplot of most freq Bi-grams
sns.set(rc={'figure.figsize':(13,8)})
h=sns.barplot(x="Bi-gram", y="Freq", data=top2_df)
h.set_xticklabels(h.get_xticklabels(), rotation=45)


#visulalize the bar plot using seaborn library - tri grams
def get_top_n3_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(3,3), 
           max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]
top3_words = get_top_n3_words(corpus, n=20)
top3_df = pd.DataFrame(top3_words)
top3_df.columns=["Tri-gram", "Freq"]
print(top3_df)
#Barplot of most freq Tri-grams
sns.set(rc={'figure.figsize':(13,8)})
j=sns.barplot(x="Tri-gram", y="Freq", data=top3_df)
j.set_xticklabels(j.get_xticklabels(), rotation=45)



#convert word matrix to integer

from sklearn.feature_extraction.text import TfidfTransformer
 
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(X)
# get feature names
feature_names=cv.get_feature_names()
 
# fetch document for which keywords needs to be extracted
doc=corpus[532]
 
#generate tf-idf for the given document
tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
tf_idf_vector.shape
print(tf_idf_vector)




#based on higest td-idf score we can extract higest scores
#Function for sorting tf_idf in descending order
from scipy.sparse import coo_matrix
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

#sort the tf-idf vectors by descending order of scores
sorted_items=sort_coo(tf_idf_vector.tocoo())
#extract only the top n; n here is 10
keywords=extract_topn_from_vector(feature_names,sorted_items,25)
 
# now print the results
print("\nText:")
print(doc)
print("\n Top Word Score:")
for k in keywords:
    print(k,keywords[k])
    
    
    #find higest weight of particular word
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
tfidf.fit(df['New Text'])
    
X_transform = tfidf.transform(df['New Text'])
#text at loc[1]
df['New Text'][1]
#find importance of words using vectorization method
print([X_transform[1,tfidf.vocabulary_['find']]])
#sentiment analysis
#!pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

#sentiment analyser
df['Sentiment Analyser'] = df.TEXT.apply(lambda x:analyser.polarity_scores(x))
#create a different column of all scores which are present as json array
 k=df['Sentiment Analyser'].apply(pd.DataFrame,index=[0]).tolist()
final_df = pd.concat(k)
final_df.index = pd.Series(final_df.index).shift(-1).fillna(0).cumsum()
#add only compound score to our original data set

df = pd.DataFrame(df)
df1=pd.DataFrame(final_df)
#add score to original data frame
df['Compound Score'] = df1['compound'].tolist()


def Category_function(x):
    if x> 0.05 :
        return "Positive"
    elif x > -0.05 and x <0.05:
        return "Neutral"
    elif x <= -0.05:
        return "Negative"

 df['Sentiment_Category']= df['Compound Score'].apply(lambda x: Category_function(x) )   
#Sentiment_Category is converted to category type
 df['Sentiment_Category'] = df['Sentiment_Category'].astype('category')
df.dtypes
#add a column to have numerical columns to category
 df['Sentiment_Category_Code'] = df['Sentiment_Category'].cat.codes
Sentiment_Category_Code= df['Sentiment_Category_Code']

 df['Sentiment_Category_code_1']= df['Sentiment_Category_Code'].apply(lambda x: 1 if x>=1 else 0)   
Sentiment_Category_code_1=df['Sentiment_Category_code_1']
#naive bayes
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Sentiment_Category_code_1, test_size=0.3)
from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB()
clf.fit(X_train, y_train)
print (clf.score(X_train, y_train))
print (clf.score(X_test, y_test))
predicted_result=clf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_test,predicted_result))
Naive_bayes_score = accuracy_score(y_test, predicted_result)
Naive_bayes_score
#Using random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# calculating the random forest accuracy 
y_pred_rf = rf.predict(X_test)
random_forest_score = accuracy_score(y_test, y_pred_rf)
random_forest_score
print(classification_report(y_test,y_pred_rf))

#SVM
from sklearn import svm
svc = svm.SVC()
svc.fit(X_train, y_train)

#calculating the SVM accyracy 
y_pred_svm = svc.predict(X_test)
svc_score = accuracy_score(y_test, y_pred_svm)
svc_score




#lstm




from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re



max_fatures = 30000
tokenizer = Tokenizer(nb_words=max_fatures, split=' ')
tokenizer.fit_on_texts(df['New Text'].values)
X2 = tokenizer.texts_to_sequences(df['New Text'].values)
X2 = pad_sequences(X2)
Y2 = pd.get_dummies(df['Sentiment_Category_code_1']).values
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2,Y2, random_state = 42)
print(X2_train.shape,Y2_train.shape)
print(X2_test.shape,Y2_test.shape)


embed_dim = 150
lstm_out = 200
model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X2.shape[1], dropout=0.2))
model.add(LSTM(lstm_out, dropout_U=0.2,dropout_W=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

history=model.fit(X2_train, Y2_train, epochs=10, batch_size=10, validation_split=0.2)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'orange', label='train loss')
plt.plot(epochs, val_loss, 'blue', label='test loss')
plt.title("Training and validation loss")
plt.legend()



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'orange', label='train accuracy')
plt.plot(epochs, val_acc, 'blue', label='test accuracy')
plt.title("Training and validaton accuracy")
plt.legend()

# print the model summary
print (model.summary())

# test the model with pretrained weights
scores = model.evaluate(X2_test, Y2_test, verbose=1)

print("Accuracy: %.2f%%" % (scores[1]*100))

score,acc = model.evaluate(X2_test, Y2_test, verbose = 2, batch_size = 10)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))