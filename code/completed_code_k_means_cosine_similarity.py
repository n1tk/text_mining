
# coding: utf-8

# In[147]:

#Sergiu Buciumas
####Hands-on Exercise on Natural Language Processing Pipeline
##This hands-on exercise is to practice what we learned on natural language processing 
##and vector space model of documents (blogs).  

###Please write Python code to do the following
###1.	Retrieve blogs from the following URL: ‘http://feeds.feedburner.com/oreilly/radar/atom’
###2.	Please complete one of the following exercises;
###Option1:  build a search engine on retrieved blogs.  The search engine receives a user’s query (one term or multiple terms)
###and ranks all the blog entries in the descending orders of relevancy to the user’s query.  
####The measure of relevancy is calculated by cosine similarity as we discussed in class
##Option2:  Conduct K-means clustering on all retrieved blogs based on the cosine similarity of blog entries. 
###Show all blogs that belong to each cluster.  Each blog will be represented by its automatically generated summary.



# In[148]:

import io, json

def save_json(filename, data):
    with io.open('{0}.json'.format(filename), 
                 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False))

def load_json(filename):
    with io.open('{0}.json'.format(filename), 
                 encoding='utf-8') as f:
        return json.load(f) 


# In[149]:

import json
import feedparser
from bs4 import BeautifulSoup
from nltk import clean_html
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy
from nltk.cluster import KMeansClusterer, GAAClusterer, cosine_distance
import nltk.corpus
import nltk.stem

FEED_URL = 'http://feeds.feedburner.com/oreilly/radar/atom'

def cleanHtml(html):
    soup = BeautifulSoup(html,"lxml")
    return soup.get_text()

fp = feedparser.parse(FEED_URL)

print ("Fetched %s entries from '%s'" % (len(fp.entries[0].title), fp.feed.title))
blog_posts = []

for e in fp.entries:
    blog_posts.append({'title': e.title, 'content'
                      : cleanHtml(e.content[0].value)}) ##extracted the title of the blog and the content only, no link.

print (blog_posts[0])

save_json('blog',blog_posts)


# In[150]:

import json
import nltk

# Download nltk packages used in this example
nltk.download('stopwords')


blog_data = load_json('blog')

# Customize your list of stopwords as needed. Here, we add common
# punctuation and contraction artifacts.

stop_words = nltk.corpus.stopwords.words('english') + [
    '.',
    ',',
    '--',
    '\'s',
    '?',
    ')',
    '(',
    ':',
    '\'',
    '\'re',
    '"',
    '-',
    '}',
    '{',
    u'—',
    ]
counter = 0
master_list = []
for post in blog_data:
    sentences = nltk.tokenize.sent_tokenize(post['content']) #will be used in k-means
    #print(sentences)
    words = [w.lower() for sentence in sentences for w in # this ones will use in k-means
             nltk.tokenize.word_tokenize(sentence) if w[0] not in stop_words][:60]
    print(words)
    master_list.append(words)
    print(master_list)

    fdist = nltk.FreqDist(words) #will be used in k-means for creating the vector

    # Basic stats

    num_words = sum([i[1] for i in fdist.items()])
    num_unique_words = len(fdist.keys())

    # Hapaxes are words that appear only once

    num_hapaxes = len(fdist.hapaxes())

    top_10_words_sans_stop_words = [w for w in fdist.items() if w[0]
                                    not in stop_words][:25]
    total_corpus = [x[0] for x in top_10_words_sans_stop_words]
    counter += 1
    print ('====|=======|=========|=======|===========|=======|=====')
    print("Blog number:", counter)
    print ('________________________________________________________')
    print(sentences)
    print ('========================================================')
    print("len of each list of words", len(words))
    print(words)
    print ('````````````````````````````````````````````````````````')


# In[151]:

len(master_list) ####checking len of the list with words for each blog, resulting in 60 because we do have 60 blogs pulled from the site


# In[152]:

master_list ##content of the list


# In[153]:

len(master_list[0]) ###checking the lenght of the list at position 0


# In[154]:

for s in master_list:    ###printing the content of all lists
    print(s)


# In[155]:

#writing the lists to local file "new_filename.txt" as each list is a line in our file by using
##writelines in python3 to iterate thru lists

with open('new_filename.txt', 'w') as f:
    f.writelines("%s\n" % l for l in master_list)

###############words from each blog are selected, 
#####################has been scheduled 60 words what represent the content/summary of the document###################


# In[156]:

import numpy
from nltk.cluster import KMeansClusterer, GAAClusterer, cosine_distance
import nltk.corpus
import nltk.stem
stemmer_func = nltk.stem.snowball.SnowballStemmer("english").stem
stopwords = set(nltk.corpus.stopwords.words('english'))


# In[157]:

def normalize_word(word):
    return stemmer_func(word.lower())


# In[158]:

def get_words(titles):
    words = set()
    for title in job_titles:
        for word in title.split():
            words.add(normalize_word(word))
    return list(words)


# In[159]:

def vectorspaced(title):
    title_components = [normalize_word(word) for word in title.split()]
    return numpy.array([
        word in title_components and not word in stop_words
        for word in words], numpy.short)


# In[160]:

title_file = open("new_filename.txt", 'r')


# In[161]:

job_titles = [line.strip() for line in title_file.readlines()]
words = get_words(job_titles)
words[0:10]


# In[162]:

[vectorspaced(title) for title in job_titles if title]


# In[163]:

len(job_titles)


# In[164]:

for title in job_titles:
    print(title)


# In[165]:

cluster = KMeansClusterer(10, cosine_distance) ###cosine distance from nltk library
cluster.cluster([vectorspaced(title) for title in job_titles if title]) #vectorizing by building numpy array from the words
classified_examples = [cluster.classify(vectorspaced(title)) for title in job_titles]


# In[166]:

for cluster_id, title in sorted(zip(classified_examples, job_titles)):
    print ("\n","Cluster number:",cluster_id, "\n", "Words extracted from the blog representing individual blog:", "\n", "\n",title)

