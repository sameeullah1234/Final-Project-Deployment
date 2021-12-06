import json
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import os
from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import TerminalFormatter

from google_play_scraper import Sort, reviews, app
#flask connectivity
from flask import Flask, render_template, url_for, request
import re
import threading
import  time


appe = Flask(__name__)

@appe.route('/')
@appe.route('/home')

def home():
       return render_template("index.html")



@appe.route('/result',methods= ['POST', 'GET'])
def result():
    output = request.form.to_dict()
    name = output["name"]
    print(name)
    s = name
    result = re.search('id=(.*)', s)
    print(result.group(1))

    "%matplotlib inline"
    "%config InlineBackend.figure_format='retina'"

    sns.set(style='whitegrid', palette='muted', font_scale=1.2)



    app_packages = [
      result.group(1)
    ]

    app_infos = []

    for ap in tqdm(app_packages):
      info = app(ap, lang='en', country='us')
      del info['comments']
      app_infos.append(info)
      
    def print_json(json_object):
      json_str = json.dumps(
        json_object, 
        indent=2, 
        sort_keys=True, 
        default=str
      )
      print(highlight(json_str, JsonLexer(), TerminalFormatter()))
      
    print_json(app_infos[0])

    app_infos_df = pd.DataFrame(app_infos)
    app_infos_df.to_csv('apps.csv', index=None, header=True)

    app_reviews = []

    for ap in tqdm(app_packages):
      for score in list(range(1, 6)):
        for sort_order in [Sort.MOST_RELEVANT, Sort.NEWEST]:
          rvs, _ = reviews(
            ap,
            lang='en',
            country='us',
            sort=sort_order,
            count= 200 if score == 3 else 100,
            filter_score_with=score
          )
          for r in rvs:
            r['sortOrder'] = 'most_relevant' if sort_order == Sort.MOST_RELEVANT else 'newest'
            r['appId'] = ap
          app_reviews.extend(rvs)

    if not app_reviews:
       c = "no reviews yet"
       print("no reviews yet")
          
    print_json(app_reviews[0])

    len(app_reviews)

    app_reviews_df = pd.DataFrame(app_reviews)
    app_reviews_df.to_csv('reviews.csv', index=None, header=True)
      ######
    reviews_df = pd.read_csv("reviews.csv")

    len(reviews_df)

    reviews_df = reviews_df[["content"]]
    reviews_df.head()

    # remove 'No Negative' or 'No Positive' from text
    reviews_df["content"] = reviews_df["content"].apply(lambda x: x.replace("No Negative", "").replace("No Positive", ""))

    
    
    from nltk.corpus import wordnet

    def get_wordnet_pos(pos_tag):
        if pos_tag.startswith('J'):
            return wordnet.ADJ
        elif pos_tag.startswith('V'):
            return wordnet.VERB
        elif pos_tag.startswith('N'):
            return wordnet.NOUN
        elif pos_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
        
    import string
    from nltk import pos_tag
    from nltk.corpus import stopwords
    from nltk.tokenize import WhitespaceTokenizer
    from nltk.stem import WordNetLemmatizer


    def clean_text(text):
        # lower text
        text = text.lower()
        # tokenize text and remove puncutation
        text = [word.strip(string.punctuation) for word in text.split(" ")]
        # remove words that contain numbers
        text = [word for word in text if not any(c.isdigit() for c in word)]
        # remove stop words
        stop = stopwords.words('english')
        text = [x for x in text if x not in stop]
        # remove empty tokens
        text = [t for t in text if len(t) > 0]
        # pos tag text
        pos_tags = pos_tag(text)
        # lemmatize text
        text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
        # remove words with only one letter
        text = [t for t in text if len(t) > 1]
        # join all
        text = " ".join(text)
        return(text)

    # clean text data
    reviews_df["content_clean"] = reviews_df["content"].apply(lambda x: clean_text(x))

    # add sentiment anaylsis columns
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    sid = SentimentIntensityAnalyzer()
    reviews_df["sentiments"] = reviews_df["content"].apply(lambda x: sid.polarity_scores(x))
    reviews_df = pd.concat([reviews_df.drop(['sentiments'], axis=1), reviews_df['sentiments'].apply(pd.Series)], axis=1)

    # add number of characters column
    reviews_df["nb_chars"] = reviews_df["content"].apply(lambda x: len(x))

    # add number of words column
    reviews_df["nb_words"] = reviews_df["content"].apply(lambda x: len(x.split(" ")))

    # create doc2vec vector columns
    from gensim.test.utils import common_texts
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(reviews_df["content_clean"].apply(lambda x: x.split(" ")))]

    # train a Doc2Vec model with our text data
    model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

    # transform each document into a vector data
    doc2vec_df = reviews_df["content_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
    doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
    reviews_df = pd.concat([reviews_df, doc2vec_df], axis=1)

    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(min_df = 10)
    tfidf_result = tfidf.fit_transform(reviews_df["content_clean"]).toarray()
    tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
    tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
    tfidf_df.index = reviews_df.index
    reviews_df = pd.concat([reviews_df, tfidf_df], axis=1)

    
    # highest positive sentiment reviews (with more than 5 words)
    reviews_df[reviews_df["nb_words"] >= 5].sort_values("pos", ascending = False)[["content", "pos"]].to_csv('pos.csv', index=None, header=True)

    # lowest negative sentiment reviews 
    reviews_df[reviews_df["nb_words"] >= 5].sort_values("neg", ascending = False)[["content", "neg"]].to_csv('neg.csv', index=None, header=True)
    

    
    
    pr = pd.read_csv("pos.csv")
    pr = pr[["pos"]]
    print(pr[pr > 0].count())


    pn = pd.read_csv("neg.csv")
    pn = pn[["neg"]]
    print(pn[pn > 0].count())


    pre = pd.read_csv("pos.csv")
    total =  (pre["pos"]).sum()
    print(total)

    nre = pd.read_csv("neg.csv")
    totale =  (nre["neg"]).sum()
    print(totale)

    if(total > totale):
      c = "The Application is useful"
      print("The Application is useful")
    else:
      c = "The Application is not useful"
      print("The Application is not useful")


    feature = pd.read_csv("apps.csv")
      
    
    feature = feature[["installs", "score", "ratings",]]
    feature.head()
    d = feature['installs'].values[0]
    print(d)
    e = feature['score'].values[0]
    print(e)
    f = feature['ratings'].values[0]
    print(f)

      
  
    
    return render_template('index.html', name = c, install = d, score = e, rating = f)
    
    

    



if __name__ == "__main__":
     appe.run(debug=True)

##################################







