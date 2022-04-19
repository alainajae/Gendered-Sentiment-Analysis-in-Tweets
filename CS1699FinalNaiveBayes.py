import pandas as pd
import numpy as np
import matplotlib as nlp
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')
import re
import string
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from math import *
from sklearn.model_selection import train_test_split
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.metrics import confusion_matrix


#tweets stores main dataset
tweets = pd.read_csv('new_train.csv')
#replaces the positive sentiment 4 with 1, for consistency.
tweets['sentiment']=tweets['sentiment'].replace(4,1)
#tweets stored as string
tweets['tweet'] = tweets['tweet'].astype('str')



#Stop Words: A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) 
#that a search engine has been programmed to ignore,
#both when indexing entries for searching and when retrieving them as the result of a search query.
nltk.download('stopwords')
stopword = set(stopwords.words('english'))


#regex to clean tweets of mentions and urls
urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
userPattern = '@[^\s]+'

#runs 
def process_tweets(tweet):
    tweet = tweet.lower()
    tweet=tweet[1:]
        # Removing all URls 
    tweet = re.sub(urlPattern,'',tweet)
        # Removing all @username.
    tweet = re.sub(userPattern,'', tweet) 


        #Remove punctuations
    tweet = tweet.translate(str.maketrans("","",string.punctuation))
        #tokenizing words
    tokens = word_tokenize(tweet)
        #tokens = [w for w in tokens if len(w)>2]
        #Removing Stop Words
    final_tokens = [w for w in tokens if w not in stopword]
        #reducing a word to its word stem 
    wordLemm = WordNetLemmatizer()
    finalwords=[]
    for w in final_tokens:
        if len(w)>1:
            word = wordLemm.lemmatize(w)
            finalwords.append(word)
    return ' '.join(finalwords)
#reintegrating processed tweets to the dataframe
tweets['processed_tweets'] = tweets['tweet'].apply(lambda x: process_tweets(x))
tweets['processed_tweets']=tweets['processed_tweets'].apply(lambda x: " ".join([w for w in x.split() if len(w)>3]))
tokenized_tweet=tweets['processed_tweets'].apply(lambda x: x.split())


#tokenize the processed tweets for use in naive bayes

token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts = cv.fit_transform(tweets['processed_tweets'].values.astype('U'))



X=text_counts
y=tweets['sentiment']
#setting the sizes of training and test datasets, randomizing them
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.21,random_state=19)



#create sklearn complement naive bayes
cnb = ComplementNB()
#fit naive bayes to classifier training data
cnb.fit(X_train, y_train)
#predict based on training data for test data.
pred = cnb.predict(X_test)
#evaluate score by cross validation
cross_cnb = cross_val_score(cnb, X, y,n_jobs = -1)

#print accuracy results based on cnb and cross validation
print("Cross Validation score = ",cross_cnb)                
print ("Train accuracy ={:.2f}%".format(cnb.score(X_train,y_train)*100))
print ("Test accuracy ={:.2f}%".format(cnb.score(X_test,y_test)*100))

#get mean accuracy across train and test data
train_acc_cnb=cnb.score(X_train,y_train)
test_acc_cnb=cnb.score(X_test,y_test)

#put prediction results and actual results in dataframes
pred_df = pd.DataFrame(pred, columns = ['prediction'])
actual_df = y_test.to_frame()

#print these... for posterity and sanity 
print(pred_df)
print(actual_df)

#since actual was shuffled as test data, prep to resort in ascending order and merge.
actual_df.reset_index(level=0, inplace =True)
#sanity print to make sure in correct order
print(actual_df)
#now merge 
df = actual_df.join(pred_df)
#check that dataframes properly merges
print(df)
#rename first column to index, reasserts after merge
tweets = tweets.rename(columns={'Unnamed: 0':'index'})
tweets['prediction'] = ''
#print column labels of df.. again sanity check
print(tweets.columns)

#create counter variables to measure subcategories
correct = 0
wrong = 0
#female false/true positive; male false/true positive
correct_female_positive = 0
wrong_female_positive = 0
correct_male_positive = 0
wrong_male_positive = 0
#female false/true negative; male false/true negative
correct_female_negative = 0
wrong_female_negative = 0
correct_male_negative = 0
wrong_male_negative = 0
#for unassigned gender
no_gender_correct = 0
no_gender_wrong = 0

count =0

#look through dataframe to isolate each tweet as correct/incorrect, positive/ negative, male/female
for index, line in df.iterrows():
    location = line['index']
    pred = line['prediction']
    actual = tweets.iloc[location, 1]
    tweets.iloc[location, 6] = line['prediction']
  

    if (pred == actual): #correct
        if(tweets.iloc[location, 4] == 'F'): #correct female
            if(actual == 0): #correct female negative
                correct_female_negative+=1
            elif(actual == 1):#correct female positve 
                correct_female_positive+=1
        
        elif(tweets.iloc[location, 4] == 'M'): #correct male
            if(actual == 0): #correct male negative
                correct_male_negative+=1
            elif(actual == 1):#correct male positve 
                correct_male_positive+=1
        else:
            no_gender_correct +=1


    else: 
        if(tweets.iloc[location, 4] == 'F'): #correct female
            if(actual == 0): #correct female negative
                wrong_female_negative+=1
            elif(actual == 1):#correct female positve 
                wrong_female_positive+=1
        elif(tweets.iloc[location, 4] == 'M'): #correct male
            if(actual == 0): #correct male negative
                wrong_male_negative+=1
            elif(actual == 1):#correct male positve 
                wrong_male_positive+=1
        else:
            no_gender_wrong +=1



#prints the gendered subset data gathered above.
print('correct_female_negative', correct_female_negative)
print('correct_female_positive', correct_female_positive)
print('correct_male_negative', correct_male_negative)
print('correct_male_positive', correct_male_positive)

print('wrong_female_negative', wrong_female_negative)
print('wrong_female_positive', wrong_female_positive)
print('wrong_male_negative', wrong_male_negative)
print('wrong_male_positive', wrong_male_positive)

print('no gender-correct', no_gender_correct)
print('no gender-incorrect', no_gender_wrong)

print('correct', correct)
print('worng', wrong)

#stores tweets df as csv
tweets.to_csv('tweet_csv.csv')
    



#creates a bar graph for accuracy
data_cnb = [train_acc_cnb,test_acc_cnb]
labels = ['Train Accuracy','Test Accuracy']
plt.xticks(range(len(data_cnb)), labels)
plt.ylabel('Accuracy')
plt.title('Accuracy plot with best parameters')
plt.bar(range(len(data_cnb)), data_cnb,color=['pink','lavender']) 
Train_acc = mpatches.Patch(color='pink', label='Train_acc')
Test_acc = mpatches.Patch(color='lavender', label='Test_acc')
plt.legend(handles=[Train_acc, Test_acc],loc='best')
plt.gcf().set_size_inches(8, 8)
#shows the bar graph
plt.show()

#Predict test data set
y_pred_cnb =cnb.predict(X_test)

#This is the confusion matrix :

print(confusion_matrix(y_test,y_pred_cnb))

#Checking performance our model with classification report
print(classification_report(y_test, y_pred_cnb))



#finally printing F1 Precision and Recall
print("F1 score ={:.2f}%".format(f1_score(y_test, y_pred_cnb, average="macro")*100))
f1_cnb=f1_score(y_test, y_pred_cnb, average="macro")

print("Precision score ={:.2f}%".format(precision_score(y_test, y_pred_cnb, average="macro")*100))
precision_cnb=precision_score(y_test, y_pred_cnb, average="macro")

print("Recall score ={:.2f}%".format(recall_score(y_test, y_pred_cnb, average="macro")*100))  
recall_cnb=recall_score(y_test, y_pred_cnb, average="macro")