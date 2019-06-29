#!/usr/bin/env python
# coding: utf-8

# # Information retrieval project
# 
# ### Project: Semantic similarity extraction using word vectors in Mahabharata dataset

# Welcome to the Text mining project of Mahabharata! In this notebook, we will use corpus of words from Mahabharata used as an input to create word vectors using word2vec, with the help of t-SNE, reduce the dimensions of the word vectors and finally use cosine similarity to analyze semantic similarities, i.e. to answer relationship questions based on the learning. The end solution of this project will be to analyze relationships and logics in the dataset. 
# 
# Model is assessed using the real facts about the data set, to benchmark the model I have compiled 23 relationship facts and will be adding few more as I build the model. For example, below are a few of the real data used to benchmark the model.
# Dhritarastra is related to Pandu, as Sahadeva is related to Nakula
# 
#     Bhima is related to Arjuna, as Ambalika is related to Ambika
#     Pandu is related to Kunti, as Dhritarashtra is related to Gandhari
#     Bhima is related to Draupadi, as Arjuna is related to Chitrangada
#     Karna is related to Kunti, as Duryodhana is related to Gandhari
#     .
#     .
#     .
# 
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ## Exploring the Data
# Run the code cell below to load necessary Python libraries.

# In[1]:


#future is the missing compatibility layer between Python 2 and Python 3. 
#It allows you to use a single, clean Python 3.x-compatible codebase to 
#support both Python 2 and Python 3 with minimal overhead.
from __future__ import absolute_import, division, print_function


# In[2]:


#word encoding
import codecs
#finds all pathnames matching a pattern, like regex
import glob
#log events for libraries
import logging
#concurrency
import multiprocessing
#dealing with operating system
import os
#pretty print, human readable
import pprint
#regular expressions
import re
#natural language toolkit
import nltk
#word 2 vec   (conda install -c anaconda gensim=1.0.1)
import gensim.models.word2vec as w2v
#dimensionality reduction
import sklearn.manifold
#math
import numpy as np
#plotting
import matplotlib.pyplot as plt
#parse dataset
import pandas as pd
#visualization (conda install -c anaconda seaborn=0.7.1)
import seaborn as sns


# In[3]:


get_ipython().run_line_magic('pylab', 'inline')


# ### Set up logging

# In[4]:


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# ## Download NLTK tokenizer models (only the first time)

# In[5]:


##stopwords like the at a an, unnecesasry
##tokenization into sentences, punkt 
##http://www.nltk.org/

nltk.download("punkt")
nltk.download("stopwords")


# ## Preparing Corpus
# ### Load all 18 books

# In[6]:


#get the book names, matching txt file
book_filenames = sorted(glob.glob("..\Mahabharata_extract-semantic-similarities_Natural-languageprocessing\input\*.txt"))
print("Found book:")
book_filenames


# ### Combine the book into one string

# In[7]:


#initialize raw unicode , we'll add all text to this file in memory
corpus_raw = u""

#for each book, read it, open it in utf 8 format, 
#add it to the raw corpus
for book_filename in book_filenames:
    print("Reading '{0}'...".format(book_filename))
    with codecs.open(book_filename, "r", "utf-8") as book_file:
        corpus_raw += book_file.read()
    print ("Corpus is now {0} characters long".format(len(corpus_raw)))
    print ()


# ### Split the corpus into sentences

# In[8]:


#tokenizastion
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# In[9]:


#tokenize into sentences
raw_sentences = tokenizer.tokenize(corpus_raw)


# In[10]:


#convert into a list of words
#remove unnnecessary, split into words, no hyphens
#list of words
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words


# In[11]:


#sentence where each word is tokenized
sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))


# In[12]:


# Example
print(raw_sentences[5])
print(sentence_to_wordlist(raw_sentences[5]))


# In[13]:


token_count = sum([len(sentence) for sentence in sentences])
print("The book corpus contains {0:,} tokens.".format(token_count))


# ### Build model

# In[14]:


# Dimensionality of the resulting word vectors.
# more dimensions, more computationally expensive to train
# but also more accurate
# more dimensions = more generalized
num_features = 500

# Minimum word count threshold.
min_word_count = 7

# Number of threads to run in parallel.
# more workers, faster we train
num_workers = multiprocessing.cpu_count()

# Context window length.
context_size = 25

# Downsample setting for frequent words.
# 0 - 1e-5 is good for this
downsampling = 1e-3

# Seed for the random number generator, to make the results reproducible.
seed = 1


# In[15]:


mahabharata2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)


# In[16]:


mahabharata2vec.build_vocab(sentences)


# In[17]:


print("Word2Vec vocabulary length:", len(mahabharata2vec.wv.vocab))


# ## word2vec training, this might take a minute

# In[18]:


mahabharata2vec.train(sentences)


# ### Save to file, can be useful later

# In[19]:


if not os.path.exists("trained"):
    os.makedirs("trained")


# In[20]:


mahabharata2vec.save(os.path.join("trained", "mahabharata2vec.w2v"))


# In[21]:


mahabharata2vec = w2v.Word2Vec.load(os.path.join("trained", "mahabharata2vec.w2v"))


# ### Compress the word vectors into 3D space using t-SNE and plot them for further analysis

# In[22]:


tsne = sklearn.manifold.TSNE(n_components=3,perplexity=15.0, n_iter=20000,random_state=0)


# In[23]:


all_word_vectors_matrix = mahabharata2vec.wv.syn0


# In[24]:


import gc
gc.collect()


# ### Train t-SNE, this could take few minute...

# In[25]:


all_word_vectors_matrix_3d = tsne.fit_transform(all_word_vectors_matrix)


# #### Plot the big picture

# In[26]:


points = pd.DataFrame(
    [
        (word, coords[0], coords[1], coords[2])
        for word, coords in [
            (word, all_word_vectors_matrix_3d[mahabharata2vec.wv.vocab[word].index])
            for word in mahabharata2vec.wv.vocab
        ]
    ],
    columns=["word", "x", "y", "z"]
)


# In[27]:


if not os.path.exists("points_output"):
    os.makedirs("points_output")
points.to_csv(os.path.join("points_output", "points_threeD.csv"), sep=',')


# In[28]:


points.head(10)


# In[29]:


sns.set_context("poster")


# # 3D plot can be viewed here https://plot.ly/~TilakD/65/

# In[30]:


points.plot.scatter("x", "y", c = "z",s=10, figsize=(12, 12))


# ### Proper noun extraction, to get a good picture of all the characters in the book.

# In[31]:


all_words = points.word
all_words_df = pd.DataFrame(all_words)
all_words_list = all_words_df['word'].values.tolist()
str_words = ' '.join(all_words_list)
#str_words


# In[32]:


essays = str_words
tokens = nltk.word_tokenize(essays)
tagged = nltk.pos_tag(tokens)
nouns = [word for word,pos in tagged if (pos == 'NNP') or (pos == 'NNPS')]
print ("Number of nouns is {}.".format(len(nouns)))

#join into a string
joined = "\n".join(nouns).encode('utf-8')
into_string = str(nouns)

#write in an csv file
#output = open("Proper_noun.csv", "w")
#output.write(joined)
#output.close()

#print as a table
noun_string = " ".join(nouns).encode('utf-8')
nouns_df = pd.DataFrame(nouns)
nouns_df.head(10)


# In[1]:


#conda install -c conda-forge wordcloud=1.2.1
from wordcloud import WordCloud
cloud = WordCloud(width=1440, height=1080).generate(noun_string)
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')


# In[34]:


number_noun = 0
noun_points = pd.DataFrame()
noun_string_to_word = sentence_to_wordlist(noun_string)
all_words = points.word

for noun_wrd in range(len(noun_string_to_word)):
    for wrd_sl_no in range(len(points)):
        if noun_string_to_word[noun_wrd] == all_words[wrd_sl_no]:
            noun_points_old = points.iloc[[wrd_sl_no]]
            noun_points = noun_points.append(noun_points_old)


# In[35]:


type(noun_points)
#print (noun_points)
noun_points.to_csv(os.path.join("points_output", "noun_points_threeD.csv"), sep=',')


# #### Scatter plot of all the proper nouns

# In[36]:


noun_points.plot.scatter("x", "y", c = "z", s=20, figsize=(12, 12))


# #### Lets zoom in to see related characters in the book

# In[37]:


def plot_region(x_bounds, y_bounds, z_bounds):
    slice = noun_points[
        (x_bounds[0] <= noun_points.x) &
        (noun_points.x <= x_bounds[1]) & 
        (y_bounds[0] <= noun_points.y) &
        (noun_points.y <= y_bounds[1]) &
        (z_bounds[0] <= noun_points.z) &
        (noun_points.z <= z_bounds[1])
    ]
    
    ax = slice.plot.scatter("x", "y", c = "z",s=35, figsize=(15, 15))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)


# In[38]:


plot_region(x_bounds=(-5, 9), y_bounds=(-8,9), z_bounds=(-5.5,2.5))


# #### Explore semantic similarities between book characters. Words closest to the given word

# In[39]:


mahabharata2vec.most_similar("Krishna")


# In[40]:


mahabharata2vec.most_similar("Arjuna")


# In[41]:


mahabharata2vec.most_similar("Karna")


# In[42]:


mahabharata2vec.most_similar("Karna")


# #### Answer relationship questions

# In[43]:


def nearest_similarity_cosmul(start1, end1, end2):
    similarities = mahabharata2vec.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    
    #print("{start1} is related to {end1} as {start2} is related to {end2}".format(**locals()))
    #write in an csv file
    output = open("Output_relations.csv", "a")
    output.write("{} is related to {} as {} is related to {}\n".format(start1,end1,start2,end2))
    output.close()
    
    return start2


# ## Read all the test cases in the test documnet.

# In[44]:


# Read test relation data
import pandas as pd
relation_data = pd.read_csv("test_relations.csv")
print ("Relation data read successfully!")
relation_data.head(5)


# ### Answer all the relationship questions and write it in a file.

# In[45]:


#Test all combinations of the above relations:
#row 0 with all other rows, then row 1 with all the rows ....
father_son_control = len([x for x in (relation_data.Son) if str(x) != 'nan']) - 2 
mother_son_control = len([x for x in (relation_data.Son2) if str(x) != 'nan']) - 2 
husband_wife_control = len([x for x in (relation_data.Wife) if str(x) != 'nan']) - 2 
brothers_control = len([x for x in (relation_data.Brothers1_1) if str(x) != 'nan']) - 2
brothers2_control = len([x for x in (relation_data.Brothers2_1) if str(x) != 'nan']) - 2


#open file for appending
output = open("Output_relations.csv", "w")
output.write("Predicted_Relationship\n") #Column name

#Finding son given father
print("Finding son given father")
for col_number in range(len(relation_data.Son)):
    if ((col_number-1) < father_son_control):
        for col_number_2 in range(len(relation_data.Son)):
            if ((col_number_2) < father_son_control):
                son_data = nearest_similarity_cosmul(relation_data.Son[col_number],
                                                     relation_data.Father[col_number],
                                                     relation_data.Father[(col_number_2)+1])

#Finding son given mother
print("Finding son given mother")
for col_number in range(len(relation_data.Son2)):
    if ((col_number-1) < mother_son_control):
        for col_number_2 in range(len(relation_data.Son2)):
            if ((col_number_2) < mother_son_control):
                son_mom_data = nearest_similarity_cosmul(relation_data.Son2[col_number],
                                                         relation_data.Mother[col_number],
                                                         relation_data.Mother[col_number_2+1])


#Finding Brother given Brother - Set 1
print("Finding Brother given Brother  - Set 1")
for col_number in range(len(relation_data.Brothers1_1)):
    if ((col_number-1) < brothers_control):
        for col_number_2 in range(len(relation_data.Brothers1_1)):
            if ((col_number_2) < (brothers_control+1)):
                nearest_similarity_cosmul(relation_data.Brothers1_1[col_number],
                                          relation_data.Brothers1_2[col_number],
                                          relation_data.Brothers1_2[col_number_2+1])

#Finding Brother given Brother - Set 2
print("Finding Brother given Brother  - Set 2")
for col_number in range(len(relation_data.Brothers2_1)):
    if ((col_number-1) < brothers2_control):
        for col_number_2 in range(len(relation_data.Brothers2_1)):
            if ((col_number_2) < (brothers2_control+1)):
                nearest_similarity_cosmul(relation_data.Brothers2_1[col_number],
                                          relation_data.Brothers2_2[col_number],
                                          relation_data.Brothers2_2[col_number_2+1])

                
print("Done!!")
output.close()


# # Generating an excel file which contains all correct answers, Should be run only once.

# In[46]:


correct_output = open("Correct_relations.csv", "w")
correct_output.write("Correct_Relationship\n") #Column name
    
father_son_control = len(relation_data.Son) - 2 
mother_son_control = len(relation_data.Son2) - 2 
husband_wife_control = len([x for x in (relation_data.Wife) if str(x) != 'nan']) - 2 
brothers_control = len([x for x in (relation_data.Brothers1_1) if str(x) != 'nan']) - 2
brothers2_control = len([x for x in (relation_data.Brothers2_1) if str(x) != 'nan']) - 2

print("Finding right son given father")
for col_number in range(len(relation_data.Son)):
    if ((col_number-1) < father_son_control):
        for col_number_2 in range(len(relation_data.Son)):
            if ((col_number_2) < father_son_control):
                correct_output.write("{} is related to {} as {} is related to {}\n".format(relation_data.Son[col_number],
                                                                                           relation_data.Father[col_number],
                                                                                           relation_data.Son[col_number_2+1],
                                                                                           relation_data.Father[col_number_2+1]))
                
                
#Finding right father given son
print("Finding right father given son")
for col_number in range(len(relation_data.Son)):
    if ((col_number-1) < father_son_control):
        for col_number_2 in range(len(relation_data.Son)):
            if ((col_number_2) < father_son_control):
                correct_output.write("{} is related to {} as {} is related to {}\n".format(relation_data.Father[col_number],
                                                                                           relation_data.Son[col_number],
                                                                                           relation_data.Father[col_number_2+1],
                                                                                           relation_data.Son[col_number_2+1]))

#Finding right son given mother
print("Finding right son given mother")
for col_number in range(len(relation_data.Son2)):
    if ((col_number-1) < mother_son_control):
        for col_number_2 in range(len(relation_data.Son2)):
            if ((col_number_2) < mother_son_control):
                correct_output.write("{} is related to {} as {} is related to {}\n".format(relation_data.Son2[col_number],
                                                                                           relation_data.Mother[col_number],
                                                                                           relation_data.Son2[col_number_2+1],
                                                                                           relation_data.Mother[col_number_2+1]))
                
#Finding right mother given son
print("Finding right mother given son")
for col_number in range(len(relation_data.Son2)):
    if ((col_number-1) < mother_son_control):
        for col_number_2 in range(len(relation_data.Son2)):
            if ((col_number_2) < mother_son_control):
                correct_output.write("{} is related to {} as {} is related to {}\n".format(relation_data.Mother[col_number],
                                                                                          relation_data.Son2[col_number],
                                                                                           relation_data.Mother[col_number_2+1],
                                                                                           relation_data.Son2[col_number_2+1]))

#Finding right Husband given Wife
print("Finding right Husband given Wife")
for col_number in range(len(relation_data.Husband)):
    if ((col_number-1) < husband_wife_control):
        for col_number_2 in range(len(relation_data.Husband)):
            if ((col_number_2) < husband_wife_control):
                correct_output.write("{} is related to {} as {} is related to {}\n".format(relation_data.Husband[col_number],
                                                                                           relation_data.Wife[col_number],
                                                                                           relation_data.Husband[col_number_2+1],
                                                                                           relation_data.Wife[col_number_2+1]))

#Finding right Wife given Husband
print("Finding right Wife given Husband")
for col_number in range(len(relation_data.Husband)):
    if ((col_number-1) < husband_wife_control):
        for col_number_2 in range(len(relation_data.Husband)):
            if ((col_number_2) < husband_wife_control):
                correct_output.write("{} is related to {} as {} is related to {}\n".format(relation_data.Wife[col_number],
                                                                                           relation_data.Husband[col_number],
                                                                                           relation_data.Wife[col_number_2+1],
                                                                                           relation_data.Husband[col_number_2+1]))

#Finding right Brother given Brother - Set 1
print("Finding right Brother given Brother  - Set 1")
for col_number in range(len(relation_data.Brothers1_1)):
    if ((col_number-1) < brothers_control):
        for col_number_2 in range(len(relation_data.Brothers1_1)):
            if ((col_number_2) < (brothers_control+1)):
                correct_output.write("{} is related to {} as {} is related to {}\n".format(relation_data.Brothers1_1[col_number],
                                                                                           relation_data.Brothers1_2[col_number],
                                                                                           relation_data.Brothers1_1[col_number_2+1],
                                                                                           relation_data.Brothers1_2[col_number_2+1]))

#Finding right Brother given Brother - Set 2
print("Finding right Brother given Brother  - Set 2")
for col_number in range(len(relation_data.Brothers2_1)):
    if ((col_number-1) < brothers2_control):
        for col_number_2 in range(len(relation_data.Brothers2_1)):
            if ((col_number_2) < (brothers2_control+1)):
                correct_output.write("{} is related to {} as {} is related to {}\n".format(relation_data.Brothers2_1[col_number],
                                                                                           relation_data.Brothers2_2[col_number],
                                                                                           relation_data.Brothers2_1[col_number_2+1],
                                                                                           relation_data.Brothers2_2[col_number_2+1]))               
                
correct_output.close()


# # Compare correct relations with the output relationship file

# In[47]:


# Read output data
predicted_data = pd.read_csv("Output_relations.csv")
print ("Predicted correct data read successfully!")

# Read correct data
correct_data = pd.read_csv("Correct_relations.csv")
print ("Correct data read successfully!")


# In[48]:


n_predict = len(predicted_data)  #number of rows in the csv file
n_correct = len(correct_data)  #number of rows in the csv file
print ("Total number of predicted relations is {}.".format(n_predict))


# In[49]:


predicted_data.head(10)


# In[50]:


correct_data.head(10)


# # Accuarcy calculation

# In[51]:


count = 0
cycles = 0
for row_number in range(n_predict):
    for correct_relationship_row_number in range(n_correct):
        if (predicted_data.Predicted_Relationship[row_number] == correct_data.Correct_Relationship[correct_relationship_row_number]):
            count +=1
    cycles +=1
    #print ("Cycle count is {} out of {} cycles.".format(cycles,n_predict))
print("Number of correct predictions is {}.".format(count))
accuracy = ((count)/(n_predict))*100
print ("Accuracy of the model is {}%.".format(accuracy))


# In[ ]:




