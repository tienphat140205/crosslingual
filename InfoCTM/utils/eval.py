import numpy as np
from scipy.special import gammaln
from collections import Counter
import itertools
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
import os
from utils.data.file_utils import split_text_word
import argparse
from tqdm import tqdm

def purity_score(y_true, y_pred):
    """Tính Purity cho đánh giá clustering."""
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def clustering_metric(labels, preds):
    """Tính các metric cho clustering bao gồm Purity và NMI."""
    metrics_func = [
        {'name': 'Purity', 'method': purity_score},
        {'name': 'NMI', 'method': metrics.cluster.normalized_mutual_info_score},
    ]
    results = {}
    for func in metrics_func:
        results[func['name']] = func['method'](labels, preds)
    return results

def evaluate_clustering(theta, labels):
    """Đánh giá clustering dựa trên theta và labels."""
    preds = np.argmax(theta, axis=1)
    return clustering_metric(labels, preds)

def compute_topic_coherence(reference_corpus, vocab, top_words, cv_type='c_v'):
    """Tính Topic Coherence (TC) sử dụng Gensim."""
    split_top_words = split_text_word(top_words)
    num_top_words = len(split_top_words[0])
    for item in split_top_words:
        assert num_top_words == len(item)
    
    split_reference_corpus = split_text_word(reference_corpus)
    dictionary = Dictionary(split_text_word(vocab))
    
    cm = CoherenceModel(texts=split_reference_corpus, dictionary=dictionary,
                        topics=split_top_words, topn=num_top_words, coherence=cv_type)
    cv_per_topic = cm.get_coherence_per_topic()
    score = np.mean(cv_per_topic)
    return cv_per_topic, score

def TC_on_wikipedia(top_word_path, cv_type='C_V'):
    """Tính TC trên Wikipedia dataset."""
    jar_dir = "evaluations"
    wiki_dir = os.path.join(".", 'datasets')
    random_number = np.random.randint(100000)
    os.system(
        f"java -jar {os.path.join(jar_dir, 'pametto.jar')} {os.path.join(wiki_dir, 'wikipedia', 'wikipedia_bd')} {cv_type} {top_word_path} > tmp{random_number}.txt")
    cv_score = []
    with open(f"tmp{random_number}.txt", "r") as f:
        for line in f.readlines():
            if not line.startswith("202"):
                cv_score.append(float(line.strip().split()[1]))
    os.remove(f"tmp{random_number}.txt")
    return cv_score, sum(cv_score) / len(cv_score)

def compute_TD(texts):
    """Tính Topic Diversity (TD)."""
    K = len(texts)
    T = len(texts[0].split())
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split())
    counter = vectorizer.fit_transform(texts).toarray()
    TF = counter.sum(axis=0)
    TD = (TF == 1).sum() / (K * T)
    return TD

def compute_topic_diversity(top_words, _type="TD"):
    """Tính Topic Diversity (TD)."""
    return compute_TD(top_words)

def TU_eva(texts):
    """Tính Topic Uniqueness (TU)."""
    K = len(texts)
    T = len(texts[0].split())
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split())
    counter = vectorizer.fit_transform(texts).toarray()
    TU = 0.0
    TF = counter.sum(axis=0)
    cnt = TF * (counter > 0)
    for i in range(K):
        TU += (1 / cnt[i][np.where(cnt[i] > 0)]).sum() / T
    TU /= K
    return TU