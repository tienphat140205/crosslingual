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
from typing import List, Union

# --- Helper Functions ---

def split_text_word(lines: Union[str, List[str]]) -> List[List[str]]:
    """
    Split text into words, handling both file paths and lists of strings.

    Args:
        lines: File path or list of text lines

    Returns:
        List of tokenized documents (list of lists of strings)
    """
    if isinstance(lines, str): # Check if input is a file path
        if not os.path.exists(lines):
            print(f"Warning: File not found at {lines}. Returning empty list.")
            return []
        try:
            with open(lines, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading file {lines}: {e}. Returning empty list.")
            return []
    # Process list of strings (or lines read from file)
    return [line.strip().split() for line in lines if line.strip()]


def load_labels_txt(path: str) -> np.ndarray:
    """
    Load label data from a text file.

    Args:
        path: Path to label file

    Returns:
        NumPy array of labels, or empty array if file not found/readable.
    """
    if not os.path.exists(path):
        print(f"Error: Label file not found at {path}. Returning empty array.")
        return np.array([])
    try:
        with open(path, 'r', encoding='utf-8') as f:
            labels = [int(line.strip()) for line in f if line.strip()]
        return np.array(labels)
    except Exception as e:
        print(f"Error reading or parsing label file {path}: {e}. Returning empty array.")
        return np.array([])

# --- Coherence Calculation ---

def _coherence(
        reference_corpus: List[List[str]], # Expects list of lists of strings
        top_words: List[List[str]],       # Expects list of lists of strings
        coherence_type='c_v',
        topn=20
    ) -> float:
    """
    Internal function to calculate topic coherence using Gensim's coherence model.

    Args:
        reference_corpus: List of tokenized documents
        top_words: List of lists of top words for each topic
        coherence_type: Type of coherence measure ('c_v', 'u_mass', etc.)
        topn: Number of top words to consider from each topic list

    Returns:
        Mean coherence score (float), or np.nan if calculation fails.
    """
    if not reference_corpus or not top_words:
        print("Warning: Empty reference corpus or top words list provided for coherence calculation.")
        return np.nan

    try:
        # Create dictionary from the reference corpus
        dictionary = Dictionary(reference_corpus)
        if not dictionary: # Check if dictionary is empty
             print("Warning: Gensim dictionary created from reference corpus is empty.")
             return np.nan

        # Ensure top_words doesn't exceed topn if topn is specified
        processed_top_words = [topic[:topn] for topic in top_words]

        cm = CoherenceModel(
            texts=reference_corpus, # Pass the pre-split corpus
            dictionary=dictionary,
            topics=processed_top_words, # Pass the pre-split top_words
            topn=topn, # Use topn words from each topic for calculation
            coherence=coherence_type,
        )
        # Get coherence per topic, handle potential errors
        cv_per_topic = cm.get_coherence_per_topic()
        score = np.mean(cv_per_topic)
    except ValueError as ve:
         print(f"ValueError during coherence calculation (often related to empty topics/vocab): {ve}")
         score = np.nan
    except Exception as e:
        print(f"Error during coherence calculation: {e}")
        score = np.nan # Return NaN or some indicator of failure
    return score

def compute_topic_coherence(reference_corpus: List[List[str]], top_words: List[List[str]], cv_type='c_v') -> float:
    """
    Compute topic coherence for a set of top words using a reference corpus.

    Args:
        reference_corpus: List of tokenized documents
        top_words: List of lists of top words for each topic
        cv_type: Type of coherence measure ('c_v', 'u_mass', etc.)

    Returns:
        Topic coherence score (float), or np.nan if calculation fails.
    """
    # Directly call the _coherence function with pre-processed lists
    return _coherence(
        reference_corpus=reference_corpus,
        top_words=top_words,
        coherence_type=cv_type
    )

# --- Diversity Calculation ---

def compute_TD(topics: List[List[str]]) -> float:
    """
    Computes Topic Diversity (TD) as the percentage of unique words
    in the top-N words lists across all topics.

    Args:
        topics: A list of lists of strings (top words for each topic).

    Returns:
        Topic Diversity score (float, 0.0 to 1.0).
    """
    if not topics:
        return 0.0
    # Assumes topics already contains the desired top-N words
    all_top_words = [word for topic in topics for word in topic]
    if not all_top_words:
        return 0.0
    # Find the number of unique words
    unique_top_words = set(all_top_words)
    # Calculate diversity as the ratio of unique words to total words
    td = len(unique_top_words) / len(all_top_words)
    return td

def compute_topic_diversity(top_words: List[List[str]], _type="TD") -> float:
    """
    Computes topic diversity. Currently only supports TD.

    Args:
        top_words: A list of lists of strings (top words for each topic).
        _type: The type of diversity metric (default: "TD").

    Returns:
        The calculated topic diversity score.
    """
    if _type == "TD":
        # Pass the list of lists directly to the compute_TD function
        TD = compute_TD(top_words)
        return TD
    else:
        raise ValueError(f"Unsupported diversity type: {_type}")

# --- Clustering Evaluation ---

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
    # Return 0.0 if contingency_matrix is empty or sums to 0
    if contingency_matrix.sum() == 0:
        return 0.0
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def clustering_metric(labels, preds):
    """
    Calculate multiple clustering evaluation metrics.

    Args:
        labels: True labels
        preds: Predicted cluster assignments

    Returns:
        Dictionary with clustering metrics (Purity, NMI)
    """
    results = dict()
    # Calculate Purity
    results['Purity'] = purity_score(labels, preds)
    # Calculate NMI, handle cases with single cluster assignments
    if len(np.unique(labels)) > 1 and len(np.unique(preds)) > 1:
         results['NMI'] = metrics.cluster.normalized_mutual_info_score(labels, preds)
    else:
         results['NMI'] = 0.0 # NMI is 0 if only one cluster or one true label
         print("Warning: NMI calculation skipped or set to 0 (only one unique label or prediction found).")
    return results


def evaluate_clustering(theta, labels):
    """
    Evaluate clustering performance using document-topic distributions.
    Cluster assignment is based on the topic with the highest probability.

    Args:
        theta: Document-topic distributions (NumPy array, shape: [n_docs, n_topics])
        labels: Ground truth labels (NumPy array, shape: [n_docs])

    Returns:
        Dictionary with clustering metrics (Purity, NMI)
    """
    if theta.shape[0] != labels.shape[0]:
        print("Error: Mismatch between number of documents in theta and labels.")
        return {'Purity': np.nan, 'NMI': np.nan}
    preds = np.argmax(theta, axis=1)
    return clustering_metric(labels, preds)

# --- Classification Evaluation ---

def _cls(train_theta, test_theta, train_labels, test_labels, gamma='scale'):
    """
    Internal function to train and evaluate an SVM classifier.

    Args:
        train_theta: Training document-topic distributions
        test_theta: Test document-topic distributions
        train_labels: Training labels
        test_labels: Test labels
        gamma: SVM gamma parameter

    Returns:
        Dictionary with classification metrics ('acc', 'macro-F1')
    """
    if len(np.unique(train_labels)) <= 1:
        print("Warning: Only one class present in training data. Classifier might not train meaningfully.")
        # Return default/failure metrics if training is not possible/meaningful
        return {'acc': 0.0, 'macro-F1': 0.0}

    try:
        clf = SVC(gamma=gamma, probability=False) # probability=True is slower if not needed
        clf.fit(train_theta, train_labels)
        preds = clf.predict(test_theta)
        return {
            'acc': accuracy_score(test_labels, preds),
            'macro-F1': f1_score(test_labels, preds, average='macro', zero_division=0) # Handle zero division
        }
    except Exception as e:
        print(f"Error during SVM classification: {e}")
        return {'acc': np.nan, 'macro-F1': np.nan}


def crosslingual_cls(train_theta_en, train_theta_cn,
                     test_theta_en, test_theta_cn,
                     train_labels_en, train_labels_cn,
                     test_labels_en, test_labels_cn):
    """
    Perform crosslingual classification experiments (Intra-lingual and Cross-lingual).

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
        ('intra_en', 'intra_cn', 'cross_en', 'cross_cn')
    """
    results = {
        'intra_en': _cls(train_theta_en, test_theta_en, train_labels_en, test_labels_en),
        'intra_cn': _cls(train_theta_cn, test_theta_cn, train_labels_cn, test_labels_cn),
        'cross_en': _cls(train_theta_cn, test_theta_en, train_labels_cn, test_labels_en), # Train CN, Test EN
        'cross_cn': _cls(train_theta_en, test_theta_cn, train_labels_en, test_labels_cn), # Train EN, Test CN
    }
    return results


def print_results(results):
    """
    Print classification results in a formatted way.

    Args:
        results: Dictionary with classification metrics (output from crosslingual_cls)
    """
    for key, val in results.items():
        print(f"\n>>> {key.upper()}")
        # Check if metrics are valid numbers before formatting
        acc_str = f"{val.get('acc', 'N/A'):.4f}" if isinstance(val.get('acc'), (int, float)) and not np.isnan(val.get('acc')) else "N/A"
        f1_str = f"{val.get('macro-F1', 'N/A'):.4f}" if isinstance(val.get('macro-F1'), (int, float)) and not np.isnan(val.get('macro-F1')) else "N/A"
        print(f"  Accuracy   : {acc_str}")
        print(f"  Macro-F1   : {f1_str}")


# --- Main Execution Block ---

if __name__ == "__main__":
    # --- Configuration ---
    dataset_name = "Amazon_Review" # Example: Change to your dataset
    model_name = "InfoCTM"       # Example: Change to your model name
    num_topics = 50              # Example: Number of topics used in the model
    num_top_words_display = 15   # Example: Number of top words in the output files (T15)

    # Construct paths
    base_output_dir = f"/mnt/MinhNV/InfoCTM/output/{dataset_name}"
    base_data_dir = f"/mnt/MinhNV/InfoCTM/data/{dataset_name}"
    mat_path = f"{base_output_dir}/{model_name}_K{num_topics}_rst.mat"

    # Paths for text data and labels
    en_top_words_path = f"{base_output_dir}/{model_name}_K{num_topics}_T{num_top_words_display}_en.txt"
    cn_top_words_path = f"{base_output_dir}/{model_name}_K{num_topics}_T{num_top_words_display}_cn.txt"
    # Vocab path is not directly used by Gensim CoherenceModel if dictionary is built from corpus
    # en_vocab_path = f"{base_data_dir}/vocab_en"
    # cn_vocab_path = f"{base_data_dir}/vocab_cn"
    en_corpus_path = f"{base_data_dir}/train_texts_en.txt" # Using train corpus for coherence
    cn_corpus_path = f"{base_data_dir}/train_texts_cn.txt" # Using train corpus for coherence
    train_labels_en_path = f"{base_data_dir}/train_labels_en.txt"
    train_labels_cn_path = f"{base_data_dir}/train_labels_cn.txt"
    test_labels_en_path = f"{base_data_dir}/test_labels_en.txt"
    test_labels_cn_path = f"{base_data_dir}/test_labels_cn.txt"

    print(f"--- Evaluating Model: {model_name}, Dataset: {dataset_name}, K={num_topics} ---")

    # --- Load Data ---
    print("\n--- Loading Data ---")
    # Load labels
    train_labels_en = load_labels_txt(train_labels_en_path)
    train_labels_cn = load_labels_txt(train_labels_cn_path)
    test_labels_en = load_labels_txt(test_labels_en_path)
    test_labels_cn = load_labels_txt(test_labels_cn_path)
    if any(arr.size == 0 for arr in [train_labels_en, train_labels_cn, test_labels_en, test_labels_cn]):
        print("Error: Failed to load one or more label files. Exiting.")
        exit()
    print("Labels loaded successfully.")

    # Load results matrix (.mat file)
    try:
        mat = loadmat(mat_path)
        train_theta_en = mat["train_theta_en"]
        train_theta_cn = mat["train_theta_cn"]
        test_theta_en = mat["test_theta_en"]
        test_theta_cn = mat["test_theta_cn"]
        print(f"Results matrix loaded successfully from {mat_path}.")
    except FileNotFoundError:
        print(f"Error: Results matrix file not found at {mat_path}. Exiting.")
        exit()
    except KeyError as e:
        print(f"Error: Key {e} not found in results matrix {mat_path}. Exiting.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred loading the .mat file: {e}. Exiting.")
        exit()

    # Load Text Data for Coherence/Diversity (List[List[str]])
    print("Loading text data for Coherence/Diversity...")
    en_top_words_list = split_text_word(en_top_words_path)
    cn_top_words_list = split_text_word(cn_top_words_path)
    en_corpus_list = split_text_word(en_corpus_path)
    cn_corpus_list = split_text_word(cn_corpus_path)
    # Check if loading was successful
    if not en_top_words_list: print(f"Warning: Could not load or parse EN top words from {en_top_words_path}")
    if not cn_top_words_list: print(f"Warning: Could not load or parse CN top words from {cn_top_words_path}")
    if not en_corpus_list: print(f"Warning: Could not load or parse EN corpus from {en_corpus_path}")
    if not cn_corpus_list: print(f"Warning: Could not load or parse CN corpus from {cn_corpus_path}")
    print("Text data loading complete.")


    # --- Classification Evaluation ---
    print("\n================= Classification =================")
    cls_results = crosslingual_cls(
        train_theta_en, train_theta_cn,
        test_theta_en, test_theta_cn,
        train_labels_en, train_labels_cn,
        test_labels_en, test_labels_cn
    )
    print_results(cls_results)

    # --- Clustering Evaluation ---
    print("\n================= Clustering =================")
    # Combine train and test for full dataset clustering evaluation
    all_theta_en = np.vstack([train_theta_en, test_theta_en])
    all_labels_en = np.concatenate([train_labels_en, test_labels_en])
    all_theta_cn = np.vstack([train_theta_cn, test_theta_cn])
    all_labels_cn = np.concatenate([train_labels_cn, test_labels_cn])

    # Evaluate EN Clustering
    if all_theta_en.size > 0 and all_labels_en.size > 0:
        clustering_en = evaluate_clustering(all_theta_en, all_labels_en)
        purity_en_str = f"{clustering_en.get('Purity', 'N/A'):.4f}" if isinstance(clustering_en.get('Purity'), (int, float)) and not np.isnan(clustering_en.get('Purity')) else "N/A"
        nmi_en_str = f"{clustering_en.get('NMI', 'N/A'):.4f}" if isinstance(clustering_en.get('NMI'), (int, float)) and not np.isnan(clustering_en.get('NMI')) else "N/A"
        print(f"EN | Purity: {purity_en_str}, NMI: {nmi_en_str}")
    else:
        print("EN | Skipping clustering evaluation (empty data).")

    # Evaluate CN Clustering
    if all_theta_cn.size > 0 and all_labels_cn.size > 0:
        clustering_cn = evaluate_clustering(all_theta_cn, all_labels_cn)
        purity_cn_str = f"{clustering_cn.get('Purity', 'N/A'):.4f}" if isinstance(clustering_cn.get('Purity'), (int, float)) and not np.isnan(clustering_cn.get('Purity')) else "N/A"
        nmi_cn_str = f"{clustering_cn.get('NMI', 'N/A'):.4f}" if isinstance(clustering_cn.get('NMI'), (int, float)) and not np.isnan(clustering_cn.get('NMI')) else "N/A"
        print(f"CN | Purity: {purity_cn_str}, NMI: {nmi_cn_str}")
    else:
        print("CN | Skipping clustering evaluation (empty data).")


    # --- Topic Coherence & Diversity Evaluation ---
    print("\n================= Topic Coherence & Diversity =================")

    # Gensim Coherence (c_v) - Requires loaded corpus and top words lists
    print("Calculating Topic Coherence (c_v)...")
    if en_corpus_list and en_top_words_list:
        en_cv = compute_topic_coherence(en_corpus_list, en_top_words_list, cv_type='c_v')
        en_cv_str = f"{en_cv:.4f}" if not np.isnan(en_cv) else "N/A"
        print(f"Gensim c_v | EN: {en_cv_str}")
    else:
        print("Skipping EN coherence calculation (missing required corpus or top words data).")

    if cn_corpus_list and cn_top_words_list:
        cn_cv = compute_topic_coherence(cn_corpus_list, cn_top_words_list, cv_type='c_v')
        cn_cv_str = f"{cn_cv:.4f}" if not np.isnan(cn_cv) else "N/A"
        print(f"Gensim c_v | CN: {cn_cv_str}")
    else:
        print("Skipping CN coherence calculation (missing required corpus or top words data).")

    # Topic Diversity (TD) - Requires loaded top words list
    print("\nCalculating Topic Diversity (TD)...")
    if en_top_words_list:
        # Use only the top N words specified by num_top_words_display for TD consistency
        en_top_words_for_td = [topic[:num_top_words_display] for topic in en_top_words_list]
        td_en = compute_topic_diversity(en_top_words_for_td)
        td_en_str = f"{td_en:.4f}" if not np.isnan(td_en) else "N/A"
        print(f"Topic Diversity | EN (Top {num_top_words_display}): {td_en_str}")
    else:
        print("Skipping EN diversity calculation (missing top words data).")

    if cn_top_words_list:
        # Use only the top N words specified by num_top_words_display for TD consistency
        cn_top_words_for_td = [topic[:num_top_words_display] for topic in cn_top_words_list]
        td_cn = compute_topic_diversity(cn_top_words_for_td)
        td_cn_str = f"{td_cn:.4f}" if not np.isnan(td_cn) else "N/A"
        print(f"Topic Diversity | CN (Top {num_top_words_display}): {td_cn_str}")
    else:
        print("Skipping CN diversity calculation (missing top words data).")

    print("\n--- Evaluation Complete ---")