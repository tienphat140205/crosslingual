import numpy as np
from collections import defaultdict, Counter
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from scipy.io import loadmat
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from tqdm import tqdm
import os
from typing import List

def split_text_word(texts):
    texts = [text.split() for text in texts]
    return texts

def _coherence(
        reference_corpus: List[str],
        vocab: List[str],
        top_words: List[str],
        coherence_type='c_v',
        topn=20
    ):
    """
    Calculate topic coherence using Gensim's coherence model.
    
    Args:
        reference_corpus: Path to reference corpus or list of documents
        vocab: Path to vocabulary or list of vocabulary items
        top_words: Top words for each topic
        coherence_type: Type of coherence measure
        topn: Number of top words to consider
        
    Returns:
        Mean coherence score
    """
    split_top_words = split_text_word(top_words)
    split_reference_corpus = split_text_word(reference_corpus)
    dictionary = Dictionary(split_text_word(vocab))

    cm = CoherenceModel(
        texts=split_reference_corpus,
        dictionary=dictionary,
        topics=split_top_words,
        topn=topn,
        coherence=coherence_type,
    )
    cv_per_topic = cm.get_coherence_per_topic()
    score = np.mean(cv_per_topic)

    return score


def _diversity(top_words: List[str]):
    """
    Calculate topic diversity based on unique words across topics.
    
    Args:
        top_words: List of top words for each topic
        
    Returns:
        Topic diversity score
    """
    num_words = 0.
    word_set = set()
    for words in top_words:
        ws = words.split()
        num_words += len(ws)
        word_set.update(ws)

    TD = len(word_set) / num_words
    return TD


def compute_topic_coherence(reference_corpus, vocab, top_words, cv_type='c_v'):
    """
    Compute topic coherence for a set of top words.
    
    Args:
        reference_corpus: Path to reference corpus or list of documents
        vocab: Path to vocabulary or list of vocabulary items
        top_words: Top words for each topic
        cv_type: Type of coherence measure
        
    Returns:
        Topic coherence score
    """
    # Directly call the _coherence function
    return _coherence(
        reference_corpus=reference_corpus,
        vocab=vocab,
        top_words=top_words,
        coherence_type=cv_type
    )


def compute_topic_diversity(top_words):
    """
    Compute topic diversity for a set of top words.
    
    Args:
        top_words: List of top words for each topic
        
    Returns:
        Topic diversity score
    """
    # Directly call the _diversity function
    return _diversity(top_words)


def split_text_word(lines):
    """
    Split text into words, handling both file paths and lists of strings.
    
    Args:
        lines: File path or list of text lines
        
    Returns:
        List of tokenized documents
    """
    if isinstance(lines, str):
        with open(lines, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    return [line.strip().split() for line in lines if line.strip()]


def purity_score(y_true, y_pred):
    """
    Calculate purity score for clustering evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted cluster assignments
        
    Returns:
        Purity score
    """
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def clustering_metric(labels, preds):
    """
    Calculate multiple clustering evaluation metrics.
    
    Args:
        labels: True labels
        preds: Predicted cluster assignments
        
    Returns:
        Dictionary with clustering metrics
    """
    metrics_func = [
        {'name': 'Purity', 'method': purity_score},
        {'name': 'NMI', 'method': metrics.cluster.normalized_mutual_info_score},
    ]
    results = dict()
    for func in metrics_func:
        results[func['name']] = func['method'](labels, preds)
    return results


def evaluate_clustering(theta, labels):
    """
    Evaluate clustering performance.
    
    Args:
        theta: Document-topic distributions
        labels: Ground truth labels
        
    Returns:
        Dictionary with clustering metrics
    """
    preds = np.argmax(theta, axis=1)
    return clustering_metric(labels, preds)


def load_labels_txt(path):
    """
    Load label data from a text file.
    
    Args:
        path: Path to label file
        
    Returns:
        NumPy array of labels
    """
    with open(path, 'r', encoding='utf-8') as f:
        return np.array([int(line.strip()) for line in f])


def _cls(train_theta, test_theta, train_labels, test_labels, gamma='scale'):
    """
    Train and evaluate an SVM classifier.
    
    Args:
        train_theta: Training document-topic distributions
        test_theta: Test document-topic distributions
        train_labels: Training labels
        test_labels: Test labels
        gamma: SVM gamma parameter
        
    Returns:
        Dictionary with classification metrics
    """
    clf = SVC(gamma=gamma)
    clf.fit(train_theta, train_labels)
    preds = clf.predict(test_theta)
    return {
        'acc': accuracy_score(test_labels, preds),
        'macro-F1': f1_score(test_labels, preds, average='macro')
    }


def crosslingual_cls(train_theta_en, train_theta_cn,
                     test_theta_en, test_theta_cn,
                     train_labels_en, train_labels_cn,
                     test_labels_en, test_labels_cn):
    """
    Perform crosslingual classification experiments.
    
    Args:
        train_theta_en: English training document-topic distributions
        train_theta_cn: Chinese training document-topic distributions
        test_theta_en: English test document-topic distributions
        test_theta_cn: Chinese test document-topic distributions
        train_labels_en: English training labels
        train_labels_cn: Chinese training labels
        test_labels_en: English test labels
        test_labels_cn: Chinese test labels
        
    Returns:
        Dictionary with classification metrics for different settings
    """
    results = {
        'intra_en': _cls(train_theta_en, test_theta_en, train_labels_en, test_labels_en),
        'intra_cn': _cls(train_theta_cn, test_theta_cn, train_labels_cn, test_labels_cn),
        'cross_en': _cls(train_theta_cn, test_theta_en, train_labels_cn, test_labels_en),
        'cross_cn': _cls(train_theta_en, test_theta_cn, train_labels_en, test_labels_cn),
    }
    return results


def print_results(results):
    """
    Print classification results in a formatted way.
    
    Args:
        results: Dictionary with classification metrics
    """
    for key, val in results.items():
        print(f"\n>>> {key.upper()}")
        print(f"  Accuracy   : {val['acc']:.4f}")
        print(f"  Macro-F1   : {val['macro-F1']:.4f}")


if __name__ == "__main__":
    mat_path = "/mnt/MinhNV/InfoCTM/output/Amazon_Review/InfoCTM_K50_rst.mat"
    data_dir = "/mnt/MinhNV/InfoCTM/data/Amazon_Review"

    train_labels_en = load_labels_txt(f"{data_dir}/train_labels_en.txt")
    train_labels_cn = load_labels_txt(f"{data_dir}/train_labels_cn.txt")
    test_labels_en = load_labels_txt(f"{data_dir}/test_labels_en.txt")
    test_labels_cn = load_labels_txt(f"{data_dir}/test_labels_cn.txt")

    mat = loadmat(mat_path)
    train_theta_en = mat["train_theta_en"]
    train_theta_cn = mat["train_theta_cn"]
    test_theta_en = mat["test_theta_en"]
    test_theta_cn = mat["test_theta_cn"]

    print("\n================= Classification =================")
    results = crosslingual_cls(
        train_theta_en, train_theta_cn,
        test_theta_en, test_theta_cn,
        train_labels_en, train_labels_cn,
        test_labels_en, test_labels_cn
    )
    print_results(results)

    print("\n================= Clustering =================")
    clustering_en = evaluate_clustering(np.vstack([train_theta_en, test_theta_en]),
                                        np.concatenate([train_labels_en, test_labels_en]))
    clustering_cn = evaluate_clustering(np.vstack([train_theta_cn, test_theta_cn]),
                                        np.concatenate([train_labels_cn, test_labels_cn]))
    print(f"EN | Purity: {clustering_en['Purity']:.4f}, NMI: {clustering_en['NMI']:.4f}")
    print(f"CN | Purity: {clustering_cn['Purity']:.4f}, NMI: {clustering_cn['NMI']:.4f}")

    print("\n================= Topic Coherence & Diversity =================")
    en_top_words_path = "/mnt/MinhNV/InfoCTM/output/Amazon_Review/InfoCTM_K50_T15_en.txt"
    cn_top_words_path = "/mnt/MinhNV/InfoCTM/output/Amazon_Review/InfoCTM_K50_T15_cn.txt"
    en_vocab_path = f"{data_dir}/vocab_en"
    cn_vocab_path = f"{data_dir}/vocab_cn"
    en_corpus_path = f"{data_dir}/train_texts_en.txt"  # or combine with test texts
    cn_corpus_path = f"{data_dir}/train_texts_cn.txt"  # or combine with test texts
    
    # Gensim Coherence
    en_cv = compute_topic_coherence(en_corpus_path, en_vocab_path, en_top_words_path, cv_type='c_v')
    cn_cv = compute_topic_coherence(cn_corpus_path, cn_vocab_path, cn_top_words_path, cv_type='c_v')
    print(f"Gensim c_v | EN: {en_cv:.4f}, CN: {cn_cv:.4f}")

    # Topic Diversity
    td_en = compute_topic_diversity(split_text_word(en_top_words_path))
    td_cn = compute_topic_diversity(split_text_word(cn_top_words_path))
    print(f"Topic Diversity | EN: {td_en:.4f}, CN: {td_cn:.4f}")