# MakeLexiconandTDM.py

import nltk, re
import pickle
import os
import getopt
import sys
from nltk.corpus import stopwords
from gensim.models import LdaModel
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary



############MAIN##############

def usage():
  print("\nThis is the usage function\n")
  print('Usage: python '+sys.argv[0]+' -t <input text file> -p <path of input text file> -s <size in docs of input text file> -f <global word frequency threshold> [-h <help>]\n')
  
freqw=2
sizef=0
  
if __name__ == '__main__':
  
  try:
    opts, args = getopt.getopt(sys.argv[1:], 't:p:s:f:h', ['textf=','pathf=','sizef=','freqw=','help'])
  except getopt.GetoptError:
    usage()
    sys.exit(2)
    
  if len(opts)==0:
    usage()
    sys.exit(2)
  for opt, arg in opts:
    if opt in ('-h', '--help'):
        usage()
        sys.exit(2)
    elif opt in ('-t', '--textf'):
        textf = arg 
    elif opt in ('-p', '--pathf'):
        pathf = arg
    elif opt in ('-s', '--sizef'):
        sizef = int(arg)
    elif opt in ('-f', '--freqw'):
        freqw = int(arg)
    else:
        usage()
        sys.exit(2)

#######

# Load basics
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
print()


# Load documents
texts=[]

file = open(pathf+"/"+textf+".txt", "r")
lines = file.readlines()
file.close()


# Initialize lemmatizer
lemmer = WordNetLemmatizer()

print("Starting")
i=0

for line in lines:
  print("Prepro. Abstract=", i)
  # Parsing text and removing unwanted characters
  line = re.findall(r"[\w]+", line)
  
  # Switching to lower case
  lcline = [element.lower() if isinstance(element, str) else element for element in line]
  
  # Eliminating word of length less than 3
  lccline=[]
  for l in lcline:
    if (len(l)>2):
      lccline.append(l)
      
  # Remove stopwords  
  words = [word for word in lccline if not word in stopwords.words()]

  # Function to test if something is a noun
  is_noun = lambda pos: pos[:2] == 'NN'

  # Tag the tokens with POS tags
  nouns = [ lemmer.lemmatize(word)
   for (word, pos) in nltk.pos_tag(words) if is_noun(pos)] 

  texts.append(nouns)
  
  # Step tracking
  i=i+1


# Frequency thresholding
dicfreq={}
for l in texts:
  for n in l:
    if n in dicfreq:
      dicfreq[n]=dicfreq[n]+1
    else:
      dicfreq[n]=1

validwords=[]
for key in dicfreq:
  if dicfreq[key] > freqw:
    validwords.append(key)

terms=len(validwords)

ttexts=[]
for t in texts:
  l = [word for word in t if word in validwords]
  ttexts.append(l)
  
print(ttexts)

print("Data structures - IN")
id2word = Dictionary(ttexts)

print("Printing lexique")
file = open(pathf+"/Lexique"+str(sizef)+"-"+str(freqw)+".txt", "w")
terms=0
for key in id2word:
  terms = terms + 1
  file.write(id2word[key]+"\n")
file.close()

print("Printing TDM")
file = open(pathf+"/TDM"+str(sizef)+"-"+str(freqw)+".txt", "w")
corpus = [id2word.doc2bow(text) for text in ttexts]
docs=len(corpus)
file.write(str(terms)+"\n")

tdm=[[0] * terms for _ in range(docs)] # Création of a sparse tdm

i=0
for l in corpus:
  for e in l:
    tdm[i][e[0]]=e[1]
  i=i+1 

for i in range(docs):
  for j in range(terms):
    if (j < terms-1): 
      file.write(str(tdm[i][j])+" ")
    else:
      file.write(str(tdm[i][j]))
  file.write("\n")
  
file.close()

print("Data structures - OUT")

# Pickle the different main structures

with open(pathf+"/Texts"+str(sizef)+"-"+str(freqw)+".pkl",'wb') as file:
  pickle.dump(ttexts, file) 
file.close()

with open(pathf+"/Lexique"+str(sizef)+"-"+str(freqw)+".pkl",'wb') as file:
  pickle.dump(id2word, file) 
file.close()

with open(pathf+"/TDM"+str(sizef)+"-"+str(freqw)+".pkl",'wb') as file:
  pickle.dump(corpus, file) 
file.close()





