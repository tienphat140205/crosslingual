import os
import torch
import numpy as np
import scipy.sparse
import torch
from torch.utils.data import Dataset, DataLoader
from utils.data import file_utils

class TextDataset(Dataset):
    def __init__(self, bow_lang1, bow_lang2, cluster_labels_lang1=None, cluster_labels_lang2=None):
        self.num_docs = min(len(bow_lang1), len(bow_lang2))
        self.bow_lang1 = bow_lang1[:self.num_docs]
        self.bow_lang2 = bow_lang2[:self.num_docs]

        if cluster_labels_lang1 is not None and cluster_labels_lang2 is not None:
            self.cluster_labels_lang1 = cluster_labels_lang1[:self.num_docs]
            self.cluster_labels_lang2 = cluster_labels_lang2[:self.num_docs]

    def __len__(self):
        return self.num_docs

    def __getitem__(self, index):
        return_dict = {
            'bow_lang1': self.bow_lang1[index],
            'bow_lang2': self.bow_lang2[index]
        }
        if self.cluster_labels_lang1 is not None and self.cluster_labels_lang2 is not None:
            return_dict['cluster_lang1'] = self.cluster_labels_lang1[index]
            return_dict['cluster_lang2'] = self.cluster_labels_lang2[index]
        return return_dict

class DatasetHandler:
    def __init__(self, dataset, batch_size, lang1, lang2):
        data_dir = f'/content/drive/MyDrive/projects/DualTopic/data/{dataset}'
        self.batch_size = batch_size
        self.lang1 = lang1
        self.lang2 = lang2
        self.train_bow_matrix_lang1 = scipy.sparse.load_npz(os.path.join(data_dir, f'train_bow_matrix_{lang1}.npz')).toarray()
        self.train_bow_matrix_lang2 = scipy.sparse.load_npz(os.path.join(data_dir, f'train_bow_matrix_{lang2}.npz')).toarray()
        self.test_bow_matrix_lang1 = scipy.sparse.load_npz(os.path.join(data_dir, f'test_bow_matrix_{lang1}.npz')).toarray()
        self.test_bow_matrix_lang2 = scipy.sparse.load_npz(os.path.join(data_dir, f'test_bow_matrix_{lang2}.npz')).toarray()
        self.cluster_labels_lang1 = np.load(os.path.join(data_dir, f'cluster_labels_{lang1}.npy'))
        self.cluster_labels_lang2 = np.load(os.path.join(data_dir, f'cluster_labels_{lang2}.npy'))
        self.train_size = len(self.train_bow_matrix_lang1)
        self.vocab_size_lang1 = self.train_bow_matrix_lang1.shape[1]
        self.vocab_size_lang2 = self.train_bow_matrix_lang2.shape[1]
        self.train_bow_matrix_lang1, self.test_bow_matrix_lang1 = self.move_to_cuda(self.train_bow_matrix_lang1, self.test_bow_matrix_lang1)
        self.train_bow_matrix_lang2, self.test_bow_matrix_lang2 = self.move_to_cuda(self.train_bow_matrix_lang2, self.test_bow_matrix_lang2)

        def normalize_bow(bow_matrix):
          doc_sums = bow_matrix.sum(axis=1, keepdim=True)
            # Thêm epsilon nhỏ để tránh chia cho 0 nếu có tài liệu rỗng
          normalized_bow = bow_matrix / (doc_sums + 1e-8)
          return normalized_bow

        # Chuẩn hóa cả train và test cho cả hai ngôn ngữ
        self.train_bow_matrix_lang1 = normalize_bow(self.train_bow_matrix_lang1)
        self.test_bow_matrix_lang1 = normalize_bow(self.test_bow_matrix_lang1)
        self.train_bow_matrix_lang2 = normalize_bow(self.train_bow_matrix_lang2)
        self.test_bow_matrix_lang2 = normalize_bow(self.test_bow_matrix_lang2)
        print("BoW normalization complete.")
        self.train_loader = DataLoader(
            TextDataset(self.train_bow_matrix_lang1, self.train_bow_matrix_lang2, 
                        self.cluster_labels_lang1, self.cluster_labels_lang2),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        self.test_loader = DataLoader(
            TextDataset(self.test_bow_matrix_lang1, self.test_bow_matrix_lang2),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )

    def move_to_cuda(self, train_bow_matrix, test_bow_matrix):
        train_bow_matrix = torch.as_tensor(train_bow_matrix).float()
        test_bow_matrix = torch.as_tensor(test_bow_matrix).float()
        if torch.cuda.is_available():
            train_bow_matrix = train_bow_matrix.cuda()
            test_bow_matrix = test_bow_matrix.cuda()
        return train_bow_matrix, test_bow_matrix

    def collate_fn(self, batch):
        bow_lang1 = torch.stack([item['bow_lang1'] for item in batch])
        bow_lang2 = torch.stack([item['bow_lang2'] for item in batch])
        if 'cluster_lang1' in batch[0]:
            cluster_info = [(item['cluster_lang1'], item['cluster_lang2']) for item in batch]
            return {
                'bow_lang1': bow_lang1,
                'bow_lang2': bow_lang2,
                'cluster_info': cluster_info
            }
        return {
            'bow_lang1': bow_lang1,
            'bow_lang2': bow_lang2
        }