import nltk, re
import pickle as pkl
import os
import getopt
import sys
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from collections import defaultdict
from gensim.corpora.dictionary import Dictionary
import json
import numpy as np

pathf = "C:/Users/medam/Desktop/SDM-Project/output"  
textf = "ArXiv400"  

############ MAIN ##############

def usage():
    print("\nThis is the usage function\n")
    print('Usage: python ' + sys.argv[0] + ' -t <generic core name of input files> -p <path of files> -n <number of topics> [-h <help>]\n')

nbtop = 10  

# Function to recursively convert numpy.float32 to float
def convert_numpy_floats(obj):
    if isinstance(obj, dict):
        return {key: convert_numpy_floats(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_floats(item) for item in obj]
    elif isinstance(obj, np.float32):  # Check if it's a numpy.float32 type
        return float(obj)  # Convert to a standard float
    return obj

# Function to calculate diversity
def calculate_diversity_score(lda, id2word, nbtop):
    # Extract top words for each topic
    top_words = []
    for i in range(nbtop):
        topic_terms = lda.get_topic_terms(i, topn=20)  # Extract the top 20 words
        top_words.append(set([id2word[word[0]] for word in topic_terms]))  # Create a set of top words for each topic

    # Calculate Jaccard similarity between all pairs of topics
    similarities = []
    for i in range(len(top_words)):
        for j in range(i + 1, len(top_words)):
            intersection = len(top_words[i].intersection(top_words[j]))
            union = len(top_words[i].union(top_words[j]))
            similarity = intersection / union  # Jaccard similarity
            similarities.append(similarity)
    
    # Diversity score is the inverse of average similarity
    avg_similarity = np.mean(similarities)
    diversity_score = 1 - avg_similarity  # Higher value indicates more diversity
    
    return diversity_score

if __name__ == "__main__":
    print("Loading lexique")
    with open(pathf + "/" + "Lexique" + textf + ".pkl", 'rb') as f:
        id2word = pkl.load(f)

    print("Loading TDM")  # Obsolete
    with open(pathf + "/" + "TDM" + textf + ".pkl", 'rb') as f:
        corpus = pkl.load(f)

    print("Loading text tokens")
    with open(pathf + "/" + "Texts" + textf + ".pkl", 'rb') as f:
        texts = pkl.load(f)

    lda = LdaModel(
        corpus,
        num_topics=nbtop,
        id2word=id2word,
        passes=1000,
        alpha='auto',
        eta='auto',
        decay=0.5,
        offset=1.0
    )

    # Initialize the results dictionary
    results = {}

    # Print topic description and store in results
    topic_descriptions = []
    for i in range(0, nbtop):
        value = lda.get_topic_terms(i)
        print("Topic ", i + 1)
        topic_terms = []
        for j in value:
            word = id2word[j[0]]
            print(f"P({word}) = {j[1]}")
            topic_terms.append({"word": word, "probability": j[1]})
        topic_descriptions.append({"topic": i + 1, "terms": topic_terms})
        print()
    results['topic_descriptions'] = topic_descriptions

    # Compute Perplexity, a measure of how good the model is (lower the better).
    perplexity_lda = lda.log_perplexity(corpus)
    print(f'Perplexity = {perplexity_lda}')
    results['perplexity'] = perplexity_lda

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda, texts=texts, dictionary=id2word, coherence='c_v', topn=20)
    coherence_lda = coherence_model_lda.get_coherence()
    print(f'Coherence = {coherence_lda}\n')
    results['coherence'] = coherence_lda

    # Calculate and print diversity score
    diversity_score = calculate_diversity_score(lda, id2word, nbtop)
    print(f"Diversity Score: {diversity_score}")

    # Compute topic proportions for each document
    doc_topics = lda.get_document_topics(corpus, minimum_probability=0)

    # Initialize a dictionary to store the most typical document for each topic
    most_typical_docs = defaultdict(lambda: (-1, -1))  # (doc_index, probability)

    # Iterate over documents
    for doc_index, doc_topic_proportions in enumerate(doc_topics):
        for topic_id, proportion in doc_topic_proportions:
            if proportion > most_typical_docs[topic_id][1]:
                most_typical_docs[topic_id] = (doc_index, proportion)

    # Print most typical documents for each topic and store in results
    typical_docs = []
    for topic_id, (doc_index, proportion) in most_typical_docs.items():
        print(f"Most typical document for topic {topic_id + 1} is document {doc_index + 1} with proportion {proportion}")
        typical_docs.append({"topic": topic_id + 1, "most_typical_document": doc_index + 1, "proportion": proportion})
    results['most_typical_documents'] = typical_docs

    # Convert results to JSON and save, ensuring numpy.float32 is handled
    results = convert_numpy_floats(results)  # Convert all float32 to float

    json_output_path = os.path.join(pathf, f"lda_results2_{textf}.json")
    with open(json_output_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Results saved to {json_output_path}")
