#### LDA for AP dataset #####

## Author: Mridula Maddukuri 

"""references: 
https://miguelmalvarez.com/2015/03/20/classifying-reuters-21578-collection-with-python-representing-the-data/       

"""

# # to check how fast the code snippet is 




#### READ DATA ####
with open('ap.txt') as f:
    docs = f.read().split('\n')
# get rid of any expression lwith length ess than 50. This gets rid of <TEXT> like expressions.
docs = [w for w in docs if len(w) > 50]


#### DOCUMENT REPRESENTATION ####
from nltk.corpus import stopwords # stop words
from nltk.tokenize import wordpunct_tokenize,word_tokenize # splits sentences into words
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk.stem.lancaster import LancasterStemmer # extract the roots of words 
from nltk.stem.porter import PorterStemmer # extract the roots of words 
import re
import lda
from copy import deepcopy
import numpy
import matplotlib.pyplot as plt 


# define stop words
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}','said','would']) # remove it if you need punctuation

# https://miguelmalvarez.com/2015/03/20/classifying-reuters-21578-collection-with-python-representing-the-data/       

def stemEachDoc(text):
    words = map(lambda word: word.lower(), word_tokenize(text))
    new_text = " ".join(map(lambda token: PorterStemmer().stem(token),words))
    return new_text

def stemAllDocs(docs):
    new_docs = deepcopy(docs)
    for i in range(len(docs)):
        new_docs[i] = stemEachDoc(docs[i])
    return new_docs

def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text)); # splitting string into words
    words = [word for word in words if word not in stop_words]
    tokens =(list(map(lambda token: PorterStemmer().stem(token),words))) # extracting the stems 
    p = re.compile('[a-zA-Z]+') # to remove numbers from text
    filtered_tokens = list(filter(lambda token:p.match(token) and len(token)>=min_length,tokens));
    return filtered_tokens

def tokenize_stemmed(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [word for word in words if word not in stop_words]
    p = re.compile('[a-zA-Z]+')
    filtered_words = list(filter(lambda token:p.match(token) and len(token)>=min_length,words))
    return filtered_words

def extractVocabulary(docs):
    # """Returns vocabulary from a list of documents"""
    word_list = []
    for doc in docs:
        new_words = tokenize(doc)
        word_list = word_list + new_words
        #print len(word_list)
    vocab = list(set(word_list))
    return vocab

def extractVocabulary_stemmed(docs):
    # """Returns vocabulary from a list of documents"""
    word_list = []
    for i in range(len(docs)):
        new_words = tokenize_stemmed(docs[i])
        word_list = word_list + new_words
        #print len(word_list)
    vocab = list(set(word_list))
    return vocab



# excluding stemming because it's not giving words that make sense
vocab = extractVocabulary_stemmed(docs)
#len(vocab) #26145 # 37173 without stemming


####### LDA LIBRARY ########
# for lda library : countVectorizer is better 
# counting the occurrences of tokens in each document
# for LDA using lda library
cv = CountVectorizer(vocabulary= vocab)
lda_X = cv.fit_transform(docs).toarray()
#fitting the LDA model for clustering our documents with 6 topics and 600 iterations
model = lda.LDA(n_topics=6, n_iter=500, random_state=1)
model.fit(lda_X)

#selecting the topic words from the fit

topic_word = model.topic_word_
n = 10
for i, topic_dist in enumerate(topic_word):
    topic_words = numpy.array(vocab)[numpy.argsort(topic_dist)][:-(n+1):-1]
    print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))




x=[]
c=0
#taking the documents 6 to 10 for giving their probability plots for different topics

for i in model.doc_topic_: # gives the probability distribution over topics for each document
    print i
    if c>=5:
        x.append(i)
    c=c+1
    if c==10:


        break
a=[]
b=[]
d=[]
e=[]
f=[]
g=[]
for i in x:
    a.append(i[0])
    b.append(i[1])
    d.append(i[2])
    e.append(i[3])
    f.append(i[4])
    g.append(i[5])
ind = numpy.arange(5)

width=0.12
fig, ax = plt.subplots()
rects1 = ax.bar(ind, a, width, color='r')


rects2 = ax.bar(ind + width, b, width, color='y')
rects3 = ax.bar(ind + 2*width, d, width, color='g')
rects4 = ax.bar(ind + 3*width, e, width, color='orange')
rects5 = ax.bar(ind + 4*width, f, width, color='b')
rects6 = ax.bar(ind + 5*width, g, width, color='black')


# add some text for labels, title and axes ticks
ax.set_ylabel('Probability')
ax.set_title('Probabilites for each topic for the first 5 documents')
ax.set_xticks(ind + width)
ax.set_xticklabels(('Doc 1', 'Doc 2', 'Doc 3', ' Doc 4', 'Doc 5'))

ax.legend((rects1[0], rects2[0],rects3[0],rects4[0],rects5[0],rects6[0]), ('Topic 1', 'Topic 2','Topic 3','Topic 4','Topic 5','Topic 6'))


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%.1f' % float(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
autolabel(rects6)
plt.show()

#plotting the leaf plot for all the words across all the topics

f, ax= plt.subplots(6, 1, figsize=(8, 6), sharex=True)
for i, k in enumerate([0, 1, 2, 3, 4,5]):
    ax[i].stem(topic_word[k,:], linefmt='b-',
               markerfmt='bo', basefmt='w-')
    ax[i].set_xlim(-50,4350)
    ax[i].set_ylim(0, 0.013)
    ax[i].set_ylabel("Prob")
    ax[i].set_title("topic {}".format(k))

ax[4].set_xlabel("word")

plt.tight_layout()
plt.show()

