import os
import numpy as np
import scipy
import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

from utils.data import file_utils


class BilingualTextDataset(Dataset):    
    def __init__(self, bow_en, bow_cn, clusterinfo_en=None, clusterinfo_cn=None):
        self.bow_en = bow_en
        self.bow_cn = bow_cn
        self.clusterinfo_en = clusterinfo_en
        self.clusterinfo_cn = clusterinfo_cn
        self.bow_size_en = len(self.bow_en)
        self.bow_size_cn = len(self.bow_cn)

    def __len__(self):
        return max(self.bow_size_en, self.bow_size_cn)

    def __getitem__(self, index):
        en_idx = index % self.bow_size_en
        cn_idx = index % self.bow_size_cn
        
        return_dict = {
            'bow_en': self.bow_en[en_idx],
            'bow_cn': self.bow_cn[cn_idx],
            'cluster_en': self.clusterinfo_en[en_idx] if self.clusterinfo_en is not None else None,
            'cluster_cn': self.clusterinfo_cn[cn_idx] if self.clusterinfo_cn is not None else None
        }

        return return_dict


class DatasetHandler:
    
    def __init__(self, dataset, batch_size, lang1, lang2, dict_path=None, device=0):

        data_dir = f'./data/{dataset}'
        # Use default dictionary path if not provided
        dict_path = './data/dict/ch_en_dict.dat'
        self.device = device
        self.batch_size = batch_size

        # Load data for both languages
        self.train_texts_en, self.test_texts_en, self.train_bow_matrix_en, self.test_bow_matrix_en, \
        self.vocab_en, self.word2id_en, self.id2word_en = self.read_data(data_dir, lang=lang1)
        
        self.train_texts_cn, self.test_texts_cn, self.train_bow_matrix_cn, self.test_bow_matrix_cn, \
        self.vocab_cn, self.word2id_cn, self.id2word_cn = self.read_data(data_dir, lang=lang2)

        # Set dimensions
        self.train_size_en = len(self.train_texts_en)
        self.train_size_cn = len(self.train_texts_cn)
        self.vocab_size_en = len(self.vocab_en)
        self.vocab_size_cn = len(self.vocab_cn)

        # Load cluster information
        self.clusterinfo_en = np.load(os.path.join(data_dir, f'cluster_labels_{lang1}_cosine.npy'))
        self.clusterinfo_cn = np.load(os.path.join(data_dir, f'cluster_labels_{lang2}_cosine.npy'))
        
        # Load translation dictionary
        self.trans_dict, self.trans_matrix_en, self.trans_matrix_cn = self.parse_dictionary(dict_path)

        # Load pre-trained word embeddings
        self.pretrain_word_embeddings_en = scipy.sparse.load_npz(
            os.path.join(data_dir, f'word2vec_{lang1}.npz')).toarray()
        self.pretrain_word_embeddings_cn = scipy.sparse.load_npz(
            os.path.join(data_dir, f'word2vec_{lang2}.npz')).toarray()
        
        # Create translation masks - these are already computed as NumPy arrays
        self.mask_en_to_cn, self.mask_cn_to_en = self.create_trans(
            self.clusterinfo_en, self.clusterinfo_cn, 
            self.train_bow_matrix_en, self.train_bow_matrix_cn
        )
        # Move data to CUDA if available
        self.clusterinfo_en, self.clusterinfo_cn = self.move_to_cuda(self.clusterinfo_en, self.clusterinfo_cn)       
        self.train_bow_matrix_en, self.test_bow_matrix_en = self.move_to_cuda(
            self.train_bow_matrix_en, self.test_bow_matrix_en)
        self.train_bow_matrix_cn, self.test_bow_matrix_cn = self.move_to_cuda(
            self.train_bow_matrix_cn, self.test_bow_matrix_cn)
        self.mask_en_to_cn, self.mask_cn_to_en = self.move_to_cuda(self.mask_en_to_cn, self.mask_cn_to_en)

        # Create data loaders
        self.train_loader = DataLoader(
            BilingualTextDataset(
                self.train_bow_matrix_en, self.train_bow_matrix_cn, 
                self.clusterinfo_en, self.clusterinfo_cn
            ), 
            batch_size=batch_size, 
            shuffle=True
        )
        
        self.test_loader = DataLoader(
            BilingualTextDataset(self.test_bow_matrix_en, self.test_bow_matrix_cn),
            batch_size=batch_size, 
            shuffle=False
        )
        
    def move_to_cuda(self, *arrays):
        results = []
        for arr in arrays:
            if isinstance(arr, np.ndarray):
                tensor = torch.tensor(arr).float()
            elif isinstance(arr, torch.Tensor):
                tensor = arr.float()
            else:
                raise TypeError("Input must be a numpy array or torch tensor")
            
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{self.device}")
                tensor = tensor.to(device)
            results.append(tensor)
        
        return results if len(results) > 1 else results[0]

    def read_data(self, data_dir, lang):
        train_texts = file_utils.read_texts(os.path.join(data_dir, f'train_texts_{lang}.txt'))
        test_texts = file_utils.read_texts(os.path.join(data_dir, f'test_texts_{lang}.txt'))
        vocab = file_utils.read_texts(os.path.join(data_dir, f'vocab_{lang}'))
        
        word2id = dict(zip(vocab, range(len(vocab))))
        id2word = dict(zip(range(len(vocab)), vocab))

        train_bow_matrix = scipy.sparse.load_npz(
            os.path.join(data_dir, f'train_bow_matrix_{lang}.npz')).toarray()
        test_bow_matrix = scipy.sparse.load_npz(
            os.path.join(data_dir, f'test_bow_matrix_{lang}.npz')).toarray()

        return train_texts, test_texts, train_bow_matrix, test_bow_matrix, vocab, word2id, id2word

    def parse_dictionary(self, dict_path):
        trans_dict = defaultdict(set)

        trans_matrix_en = np.zeros((self.vocab_size_en, self.vocab_size_cn), dtype='int32')
        trans_matrix_cn = np.zeros((self.vocab_size_cn, self.vocab_size_en), dtype='int32')

        dict_texts = file_utils.read_texts(dict_path)

        for line in dict_texts:
            terms = line.strip().split()
            if len(terms) == 2:
                cn_term, en_term = terms
                
                if cn_term in self.word2id_cn and en_term in self.word2id_en:
                    trans_dict[cn_term].add(en_term)
                    trans_dict[en_term].add(cn_term)
                    
                    cn_term_id = self.word2id_cn[cn_term]
                    en_term_id = self.word2id_en[en_term]

                    trans_matrix_en[en_term_id][cn_term_id] = 1
                    trans_matrix_cn[cn_term_id][en_term_id] = 1

        return trans_dict, trans_matrix_en, trans_matrix_cn
    
    def compute_cluster_tfidf(self, bow_matrix, cluster_labels, n_top_words=10):
        # Get unique clusters
        unique_clusters = np.unique(cluster_labels)
        result = {}
        
        for cluster_id in unique_clusters:
            # Get documents in this cluster
            cluster_docs = bow_matrix[cluster_labels == cluster_id]
            
            if len(cluster_docs) == 0:
                continue
                
            # Sum word occurrences across all documents in cluster
            cluster_word_sum = np.sum(cluster_docs, axis=0)
            
            # Calculate document frequency (in how many documents each word appears)
            doc_freq = np.sum(bow_matrix > 0, axis=0)
            doc_freq = np.maximum(doc_freq, 1)  # Avoid division by zero
            
            # TF-IDF calculation
            n_docs = bow_matrix.shape[0]
            tfidf_scores = cluster_word_sum * np.log(n_docs / doc_freq)
            
            # Get indices of top words
            if n_top_words:
                top_indices = np.argsort(-tfidf_scores)[:n_top_words]
            else:
                # Get all non-zero scores if n_top_words is None
                non_zero_indices = np.where(tfidf_scores > 0)[0]
                top_indices = non_zero_indices[np.argsort(-tfidf_scores[non_zero_indices])]
            
            # Store result for this cluster, only indices and scores
            result[int(cluster_id)] = {
                'indices': top_indices.tolist(),
                'scores': tfidf_scores[top_indices].tolist()
            }
            
        return result
    
    def get_cluster_top_words(self, lang='en', n_top_words=10):
        if lang == 'en':
            return self.compute_cluster_tfidf(
                self.train_bow_matrix_en, 
                self.clusterinfo_en,
                n_top_words
            )
        elif lang == 'cn':
            return self.compute_cluster_tfidf(
                self.train_bow_matrix_cn, 
                self.clusterinfo_cn,
                n_top_words
            )
        else:
            raise ValueError(f"Unsupported language: {lang}")
    
    def create_trans(self, clusterinfo_en, clusterinfo_cn, bow_matrix_en, bow_matrix_cn):
        """
        Create translation masks linking words between languages based on cluster membership.
        """
        # Get top words for each cluster using existing methods
        en_cluster_words = self.get_cluster_top_words(lang='en', n_top_words=10)
        cn_cluster_words = self.get_cluster_top_words(lang='cn', n_top_words=10)
        
        # Create translation masks
        mask_en_to_cn = np.zeros((self.vocab_size_en, self.vocab_size_cn))
        mask_cn_to_en = np.zeros((self.vocab_size_cn, self.vocab_size_en))
        
        # For each cluster, link words across languages
        unique_clusters = set(en_cluster_words.keys()) | set(cn_cluster_words.keys())
        
        for cluster_id in unique_clusters:
            # Check if cluster exists in both languages
            if cluster_id in en_cluster_words and cluster_id in cn_cluster_words:
                # Get word indices for this cluster
                en_indices = en_cluster_words[cluster_id]['indices']
                cn_indices = cn_cluster_words[cluster_id]['indices']
                
                # Create links between all words in the same cluster
                for en_idx in en_indices:
                    for cn_idx in cn_indices:
                        mask_en_to_cn[en_idx, cn_idx] = 1
                        mask_cn_to_en[cn_idx, en_idx] = 1
        
        return mask_en_to_cn, mask_cn_to_en