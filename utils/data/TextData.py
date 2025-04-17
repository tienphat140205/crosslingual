import os
import torch
import numpy as np
import scipy.sparse
# import torch # Đã import ở trên
from torch.utils.data import Dataset, DataLoader
# from utils.data import file_utils # Không thấy dùng trong đoạn code này

class TextDataset(Dataset):
    def __init__(self, bow_lang1, bow_lang2, cluster_labels_lang1=None, cluster_labels_lang2=None):
        # Xác định số lượng tài liệu dựa trên kích thước nhỏ nhất của bow vectors
        # và đảm bảo cả hai bow vectors có cùng số lượng tài liệu được sử dụng
        self.num_docs = min(len(bow_lang1), len(bow_lang2))
        self.bow_lang1 = bow_lang1[:self.num_docs]
        self.bow_lang2 = bow_lang2[:self.num_docs]

        # --- SỬA ĐỔI Ở ĐÂY ---
        # Luôn tạo thuộc tính self.cluster_labels_lang1 và self.cluster_labels_lang2.
        # Gán giá trị slice nếu tham số đầu vào không phải None, ngược lại gán None.
        self.cluster_labels_lang1 = cluster_labels_lang1[:self.num_docs] if cluster_labels_lang1 is not None else None
        self.cluster_labels_lang2 = cluster_labels_lang2[:self.num_docs] if cluster_labels_lang2 is not None else None
        # --- KẾT THÚC SỬA ĐỔI ---

        # (Tùy chọn) Thêm kiểm tra độ dài nếu cần thiết và cluster labels được cung cấp
        # if self.cluster_labels_lang1 is not None:
        #     assert len(self.cluster_labels_lang1) == self.num_docs, \
        #            f"Length mismatch: bow1 ({self.num_docs}) vs cluster1 ({len(cluster_labels_lang1)})"
        # if self.cluster_labels_lang2 is not None:
        #     assert len(self.cluster_labels_lang2) == self.num_docs, \
        #            f"Length mismatch: bow2 ({self.num_docs}) vs cluster2 ({len(cluster_labels_lang2)})"


    def __len__(self):
        return self.num_docs

    def __getitem__(self, index):
        # Lấy dữ liệu bow cho chỉ mục được yêu cầu
        # Đảm bảo index nằm trong phạm vi hợp lệ (mặc dù DataLoader thường xử lý việc này)
        if index >= self.num_docs:
             raise IndexError(f"Index {index} out of bounds for dataset with size {self.num_docs}")

        return_dict = {
            'bow_lang1': self.bow_lang1[index],
            'bow_lang2': self.bow_lang2[index]
        }

        # Kiểm tra này bây giờ sẽ hoạt động chính xác vì các thuộc tính luôn tồn tại
        if self.cluster_labels_lang1 is not None and self.cluster_labels_lang2 is not None:
            # Thêm cluster labels vào dictionary nếu chúng tồn tại
            # Việc cắt [:self.num_docs] trong __init__ đảm bảo index hợp lệ ở đây
            return_dict['cluster_lang1'] = self.cluster_labels_lang1[index]
            return_dict['cluster_lang2'] = self.cluster_labels_lang2[index]

        return return_dict

# Lớp DatasetHandler giữ nguyên như bạn đã cung cấp
class DatasetHandler:
    def __init__(self, dataset, batch_size, lang1, lang2):
        data_dir = f'/content/drive/MyDrive/projects/DualTopic/data/{dataset}'
        self.batch_size = batch_size
        self.lang1 = lang1
        self.lang2 = lang2
        # --- Tải dữ liệu ---
        try:
            self.train_bow_matrix_lang1 = scipy.sparse.load_npz(os.path.join(data_dir, f'train_bow_matrix_{lang1}.npz')).toarray()
            self.train_bow_matrix_lang2 = scipy.sparse.load_npz(os.path.join(data_dir, f'train_bow_matrix_{lang2}.npz')).toarray()
            self.test_bow_matrix_lang1 = scipy.sparse.load_npz(os.path.join(data_dir, f'test_bow_matrix_{lang1}.npz')).toarray()
            self.test_bow_matrix_lang2 = scipy.sparse.load_npz(os.path.join(data_dir, f'test_bow_matrix_{lang2}.npz')).toarray()
            # Giả sử cluster labels chỉ dành cho tập train, hoặc cần được chia tách nếu dùng cho cả test
            self.cluster_labels_lang1 = np.load(os.path.join(data_dir, f'cluster_labels_{lang1}.npy'))
            self.cluster_labels_lang2 = np.load(os.path.join(data_dir, f'cluster_labels_{lang2}.npy'))
        except FileNotFoundError as e:
            print(f"Lỗi: Không tìm thấy tệp dữ liệu - {e}")
            print("Hãy chắc chắn rằng các tệp .npz và .npy tồn tại trong thư mục:", data_dir)
            raise e

        self.train_size = len(self.train_bow_matrix_lang1)
        self.vocab_size_lang1 = self.train_bow_matrix_lang1.shape[1]
        self.vocab_size_lang2 = self.train_bow_matrix_lang2.shape[1]

        # --- Chuyển sang Tensor và CUDA ---
        self.train_bow_matrix_lang1, self.test_bow_matrix_lang1 = self.move_to_cuda(self.train_bow_matrix_lang1, self.test_bow_matrix_lang1)
        self.train_bow_matrix_lang2, self.test_bow_matrix_lang2 = self.move_to_cuda(self.train_bow_matrix_lang2, self.test_bow_matrix_lang2)

        # --- Định nghĩa hàm chuẩn hóa nội bộ ---
        def normalize_bow(bow_matrix):
            # Tính tổng của các từ trong mỗi tài liệu
            doc_sums = bow_matrix.sum(axis=1, keepdim=True)
            # Thêm một giá trị epsilon nhỏ để tránh chia cho 0 đối với các tài liệu rỗng
            normalized_bow = bow_matrix / (doc_sums + 1e-8)
            return normalized_bow

        # --- Chuẩn hóa BoW ---
        self.train_bow_matrix_lang1 = normalize_bow(self.train_bow_matrix_lang1)
        self.test_bow_matrix_lang1 = normalize_bow(self.test_bow_matrix_lang1)
        self.train_bow_matrix_lang2 = normalize_bow(self.train_bow_matrix_lang2)
        self.test_bow_matrix_lang2 = normalize_bow(self.test_bow_matrix_lang2)
        print("BoW normalization complete.")

        # --- Tạo DataLoader ---
        # Lưu ý: Đảm bảo self.cluster_labels_lang1/2 có cùng số lượng mẫu như train_bow nếu chúng được dùng
        # TextDataset sẽ tự động cắt bớt nếu cần thiết dựa trên self.num_docs
        self.train_loader = DataLoader(
            TextDataset(self.train_bow_matrix_lang1, self.train_bow_matrix_lang2,
                        self.cluster_labels_lang1, self.cluster_labels_lang2),
            batch_size=batch_size,
            shuffle=True, # Nên shuffle=True cho tập huấn luyện
            collate_fn=self.collate_fn,
            # Thêm num_workers và pin_memory để tăng tốc độ nếu có thể
            # num_workers=4,
            # pin_memory=True if torch.cuda.is_available() else False
        )

        # Test loader không cần cluster labels (trừ khi bạn có mục đích đặc biệt)
        self.test_loader = DataLoader(
            TextDataset(self.test_bow_matrix_lang1, self.test_bow_matrix_lang2), # Không truyền cluster labels
            batch_size=batch_size,
            shuffle=False, # Nên shuffle=False cho tập kiểm thử/đánh giá
            collate_fn=self.collate_fn,
            # num_workers=4,
            # pin_memory=True if torch.cuda.is_available() else False
        )

    def move_to_cuda(self, train_bow_matrix, test_bow_matrix):
        # Chuyển đổi NumPy arrays sang PyTorch tensors kiểu float
        train_bow_tensor = torch.as_tensor(train_bow_matrix).float()
        test_bow_tensor = torch.as_tensor(test_bow_matrix).float()
        # Chuyển tensors lên GPU nếu có CUDA
        if torch.cuda.is_available():
            train_bow_tensor = train_bow_tensor.cuda()
            test_bow_tensor = test_bow_tensor.cuda()
            print("Data moved to CUDA.")
        else:
             print("CUDA not available. Using CPU.")
        return train_bow_tensor, test_bow_tensor

    def collate_fn(self, batch):
        # Gom các tensor bow từ các mẫu trong batch
        bow_lang1 = torch.stack([item['bow_lang1'] for item in batch])
        bow_lang2 = torch.stack([item['bow_lang2'] for item in batch])

        # Tạo dictionary kết quả cơ bản
        collated_batch = {
            'bow_lang1': bow_lang1,
            'bow_lang2': bow_lang2
        }

        # Kiểm tra xem mẫu đầu tiên trong batch có cluster labels không
        # (Giả định rằng tất cả các mẫu trong batch đều có hoặc không có cùng lúc)
        if 'cluster_lang1' in batch[0] and 'cluster_lang2' in batch[0]:
            # Gom thông tin cluster thành một list các tuple hoặc xử lý theo cách khác nếu cần
            cluster_info = [(item['cluster_lang1'], item['cluster_lang2']) for item in batch]
            collated_batch['cluster_info'] = cluster_info # Hoặc trả về tensor nếu cluster labels là số
            # Nếu cluster_lang1 và cluster_lang2 là tensor số nguyên:
            # cluster_lang1 = torch.stack([item['cluster_lang1'] for item in batch])
            # cluster_lang2 = torch.stack([item['cluster_lang2'] for item in batch])
            # collated_batch['cluster_lang1'] = cluster_lang1
            # collated_batch['cluster_lang2'] = cluster_lang2


        return collated_batch