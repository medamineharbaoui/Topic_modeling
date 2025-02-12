import json
import nltk, re
import pickle as pkl
import os
import getopt
import sys
import numpy as np
from sklearn.metrics import pairwise_distances
import pandas as pd 
from hdbscan import HDBSCAN
from umap import UMAP
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_dataset
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from collections import defaultdict
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
import multiprocessing

############MAIN##############
def usage():
  print("\nThis is the usage function\n")
  print('Usage: python '+sys.argv[0]+' -s <size of subset> -p <path of files> -n <number of top words> [-h <help>]\n')

nbtow=20
subsz=400

def calculate_coherence_score(topic_model, docs, nbotw):
    # Preprocess documents
    cleaned_docs = topic_model._preprocess_text(docs)

    # Extract vectorizer and tokenizer from BERTopic
    vectorizer = topic_model.vectorizer_model
    tokenizer = vectorizer.build_tokenizer()

    # Extract features for Topic Coherence evaluation
    words = vectorizer.get_feature_names_out()
    tokens = [tokenizer(doc) for doc in cleaned_docs]
    dictionary = Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    
    # Create topic words
    all_topics = topic_model.get_topics()
    topic_words=[]
    for key in all_topics: 
      l=[]
      for t in all_topics[key]:  
        l.append(t[0])
      topic_words.append(l)

    coherence_model = CoherenceModel(topics=topic_words,
                                   texts=tokens,
                                   corpus=corpus,
                                   dictionary=dictionary,
                                   coherence='c_v',
                                   topn=nbtow)
    coherence = coherence_model.get_coherence()
    
    return coherence


def calculate_diversity_score(topic_model):
    # Get the topics and their top words
    all_topics = topic_model.get_topics()
    
    # Extract the top words for each topic
    top_words = [set([word[0] for word in all_topics[key]]) for key in all_topics]
    
    # Calculate Jaccard similarity between all pairs of topic top words
    similarities = []
    for i in range(len(top_words)):
        for j in range(i + 1, len(top_words)):
            intersection = len(top_words[i].intersection(top_words[j]))
            union = len(top_words[i].union(top_words[j]))
            similarity = intersection / union  # Jaccard similarity
            similarities.append(similarity)
    
    # The diversity score is the inverse of average Jaccard similarity
    avg_similarity = np.mean(similarities)
    diversity_score = 1 - avg_similarity  # Higher score indicates more diversity
    
    return diversity_score


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')  # Set the method for starting new processes

    try:
        opts, args = getopt.getopt(sys.argv[1:], 's:p:n:h', ['textf=','pathf=','nbtow=','help'])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
        
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit(2)
        elif opt in ('-s', '--subsz'):
            subsz = int(arg)
        elif opt in ('-p', '--pathf'):
            pathf = arg
        elif opt in ('-n', '--nbtow'):
            nbtow = int(arg)
        else:
            usage()
            sys.exit(2)

    dataset = load_dataset("CShorten/ML-ArXiv-Papers")["train"]
    abstracts = dataset["abstract"]

    # Save dataset as dataset.csv
    df = pd.DataFrame({"text": abstracts})
    df.to_csv("dataset.csv", index=False)
    print("✅ dataset.csv saved successfully!")

    # Make an abstract subset of size subz
    ssabstracts = []
    for i in range(subsz):
        ssabstracts.append(abstracts[i])

    print("Document embedding")
    # Pre-calculate embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(ssabstracts, show_progress_bar=True)
    print("Docs embedding desc. : ", np.shape(embeddings))

    print("Embedding space reduction")
    # Embedding space dimensionnality reduction
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

    print("Clustering embedding space")
    # Clustering the embedding space
    hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

    print("Document vectorization")
    # Clustering the embedding space
    vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 1))   # Extraction of single words

    # Building model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        top_n_words=nbtow,
        verbose=True
    )

    topics, probs = topic_model.fit_transform(ssabstracts, embeddings)

    # Calculate and print the coherence score
    coherence_score = calculate_coherence_score(topic_model, ssabstracts, nbtow)
    print("Coherence=", coherence_score)

    # Calculate and print the diversity score
    diversity_score = calculate_diversity_score(topic_model)
    print(f"Diversity Score: {diversity_score}")

    # Extract top 20 words from the topics
    top_words_dict = {}

    # Get the topics
    all_topics = topic_model.get_topics()

    # Extract top 20 words from the first 10 topics
    for i, topic in enumerate(all_topics):
        if i >= 10:  # Limit to the first 10 topics
            break
        top_words = [word[0] for word in all_topics[topic][:nbtow]]  # Get top 20 words for the current topic
        top_words_dict[f"Topic_{i+1}"] = top_words

    # Print the top 20 words for each topic
    for topic, words in top_words_dict.items():
        print(f"{topic}: {', '.join(words)}")

    # Save the top words to a JSON file
    with open("top_words.json", "w") as json_file:
        json.dump(top_words_dict, json_file, indent=4)

    print("✅ Top 20 words from the first 10 topics saved to 'top_words.json' successfully!")
