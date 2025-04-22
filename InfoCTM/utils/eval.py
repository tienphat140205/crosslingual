import numpy as np
from collections import defaultdict
from sklearn import metrics
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import os
from sklearn.feature_extraction.text import CountVectorizer
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


def split_text_word(lines):
    if isinstance(lines, str):
        with open(lines, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    return [line.strip().split() for line in lines if line.strip()]


def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def clustering_metric(labels, preds):
    metrics_func = [
        {'name': 'Purity', 'method': purity_score},
        {'name': 'NMI', 'method': metrics.cluster.normalized_mutual_info_score},
    ]
    results = dict()
    for func in metrics_func:
        results[func['name']] = func['method'](labels, preds)
    return results


def evaluate_clustering(theta, labels):
    preds = np.argmax(theta, axis=1)
    return clustering_metric(labels, preds)


def compute_topic_coherence(reference_corpus, vocab, top_words, cv_type='c_v'):
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
    K = len(texts)  # Number of topic
    T = len(texts[0])  # Number of words in each topic (assuming all topics have the same length)
    
    # Convert lists of words to space-separated strings for CountVectorizer
    topic_strings = [' '.join(words) for words in texts]
    vectorizer = CountVectorizer()
    counter = vectorizer.fit_transform(topic_strings).toarray()
    TF = counter.sum(axis=0)
    TD = (TF == 1).sum() / (K * T)
    return TD


def compute_topic_diversity(top_words):
    return compute_TD(top_words)


def load_labels_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        return np.array([int(line.strip()) for line in f])


def _cls(train_theta, test_theta, train_labels, test_labels, gamma='scale'):
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

    results = {
        'intra_en': _cls(train_theta_en, test_theta_en, train_labels_en, test_labels_en),
        'intra_cn': _cls(train_theta_cn, test_theta_cn, train_labels_cn, test_labels_cn),
        'cross_en': _cls(train_theta_cn, test_theta_en, train_labels_cn, test_labels_en),
        'cross_cn': _cls(train_theta_en, test_theta_cn, train_labels_en, test_labels_cn),
    }
    return results


def print_results(results):
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
    _, en_cv = compute_topic_coherence(en_corpus_path, en_vocab_path, en_top_words_path, cv_type='c_v')
    _, cn_cv = compute_topic_coherence(cn_corpus_path, cn_vocab_path, cn_top_words_path, cv_type='c_v')
    print(f"Gensim c_v | EN: {en_cv:.4f}, CN: {cn_cv:.4f}")

    # Topic Diversity
    td_en = compute_topic_diversity(split_text_word(en_top_words_path))
    td_cn = compute_topic_diversity(split_text_word(cn_top_words_path))
    print(f"Topic Diversity | EN: {td_en:.4f}, CN: {td_cn:.4f}")
