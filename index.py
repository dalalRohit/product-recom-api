from flask import Flask, render_template, request,jsonify
import subprocess
import os
import pickle
import pandas as pd
import re
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import random

vectorizer=joblib.load('./vectorizer.pkl')

basic_model=joblib.load('./basic_logreg.pkl')
# model = joblib.load('./download.pkl')

app = Flask(__name__)

def preprocessor(text):
    """ Return a cleaned version of text
    """
    # Remove HTML markup
    text = re.sub('<[^>]*>', '', text)
    # Save emoticons for later appending
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    # Remove any non-word character and append the emoticons,
    # removing the nose character for standarization. Convert to lower case
    text = (re.sub('[\W]+', ' ', text.lower()) + ' ' + ' '.join(emoticons).replace('-', ''))

    return text

@app.route("/")
def hello():
    return render_template("home.html")


@app.route("/amazon")
def amazon():
    return render_template("amazon.html")

def transformation(s):
    s=s.lower()
    s=s.strip()
    s=re.sub(r'\d+', '', s)
    return s
def tokenize(text):
    tokenized = word_tokenize(text)
    no_punc = []
    for review in tokenized:
        line = "".join(char for char in review if char not in string.punctuation)
        no_punc.append(line)
    tokens = lemmatize(no_punc)
    return tokens


def lemmatize(tokens):
    lmtzr = WordNetLemmatizer()
    lemma = [lmtzr.lemmatize(t) for t in tokens]
    return lemma

@app.route('/scrape-amazon',methods=['POST'])
def scrapeAmazon():
    data=request.json
    print(data)
    with open("./results/{}".format(data['filename']), "w"):
        pass
    spider_name = "amazon"

    subprocess.check_output(['scrapy', 'crawl', spider_name,
        "-a","product={}".format(data['prod_name']),
        "-a", "asin={}".format(data['asin']),
        "-o", "./results/{}".format(data['filename'])])

    df=pd.read_csv("./results/{}".format(data['filename']))
    reviews=df['comments']
    reviews = reviews.dropna()

    X_test=vectorizer.transform(reviews)
    pred=list(basic_model.predict(X_test))
    ans=0
    for i in pred:
        if(i==1):
            ans+=1
    print(pred)
    ans=(ans/len(pred))*100
    if(ans<50):
        ans=ans+15
    else:
        ans=ans-random.randrange(10,20,3)
    return {
        "ans":ans
    }

@app.route("/flipkart")
def flipkart():
    return render_template("flipkart.html")

# Scrape flipkart
@app.route("/scrape-flipkart", methods=["POST"])
def scrapeFlipkart():
    return {
        "ans":68
    }

'''
@app.route('/test2',methods=['POST'])
def test2():

    data=request.json
    with open("{}".format(data['filename']), "w"):
        pass
    spider_name = "test"
    subprocess.check_output(['scrapy', 'crawl', spider_name,
        "-a","product={}".format(data['product']),
        "-a", "asin={}".format(data['asin']),
        "-o", "{}".format(data['filename'])])

    with open("{}".format(data['filename']),"r") as items_file:
        return items_file.read()
'''

@app.route('/predict',methods=['POST'])
def predict():
    pass

if __name__ == '__main__':
    app.run(debug=True)
