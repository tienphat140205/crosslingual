import os
import numpy as np
from cuml.cluster import HDBSCAN
from sklearn.preprocessing import normalize
from scipy.linalg import svd
from scipy.stats import ttest_ind

# Đường dẫn gốc tới thư mục chứa data
base_data_dir = '/content/drive/MyDrive/projects/DualTopic/data'

# Dictionary định nghĩa các dataset và ngôn ngữ không phải tiếng Anh
datasets = {
    'Amazon_Review': 'cn',  # Tiếng Trung
    'ECNews': 'cn',         # Tiếng Trung
    'Rakuten_Amazon': 'ja'  # Tiếng Nhật
}

# Tham số HDBSCAN (đã điều chỉnh để giảm nhiễu)
min_cluster_size = 5   # Giảm để cho phép cluster nhỏ hơn
min_samples = 3        # Giảm để dễ tạo cluster hơn
cluster_selection_method = 'leaf'  # Đổi sang 'leaf' để tạo nhiều cluster nhỏ hơn

# Tham số giảm chiều
n_components = 100  # Số chiều sau khi giảm (theo tài liệu, giảm từ 768 xuống 100)

# Phương pháp tinh chỉnh chiều: 'u-SVD' hoặc 'SVD-LR'
dimension_refinement_method = 'u-SVD'  # Có thể đổi sang 'SVD-LR'

# Vòng lặp qua từng dataset
for dataset, lang in datasets.items():
    print(f"Đang xử lý {dataset}...")

    # Tạo đường dẫn tới file embedding
    en_train_path = os.path.join(base_data_dir, dataset, 'doc_embeddings_en_train.npy')
    lang_train_path = os.path.join(base_data_dir, dataset, f'doc_embeddings_{lang}_train.npy')

    # Kiểm tra xem file có tồn tại không
    if not os.path.exists(en_train_path) or not os.path.exists(lang_train_path):
        print(f"Lỗi: Không tìm thấy file embedding cho {dataset}")
        continue

    # Load embedding
    doc_embeddings_en = np.load(en_train_path)
    doc_embeddings_lang = np.load(lang_train_path)

    # Ghép embedding của 2 ngôn ngữ thành 1 mảng
    all_embeddings = np.vstack([doc_embeddings_en, doc_embeddings_lang])

    # Chuẩn hóa embedding để độ dài = 1 (norm L2 = 1)
    print("Chuẩn hóa embedding...")
    all_embeddings_normalized = normalize(all_embeddings, axis=1, norm='l2')

    # Bước tinh chỉnh chiều bằng SVD (u-SVD hoặc SVD-LR)
    print("Tinh chỉnh chiều bằng SVD...")
    # Thực hiện SVD: E = U * Sigma * V^T
    U, Sigma, Vt = svd(all_embeddings_normalized, full_matrices=False)
    
    # Chỉ giữ lại n_components chiều
    U = U[:, :n_components]  # U: ma trận trái (m x n_components)
    Sigma = np.diag(Sigma[:n_components])  # Sigma: ma trận chéo (n_components x n_components)
    Vt = Vt[:n_components, :]  # V^T: ma trận phải (n_components x d)

    if dimension_refinement_method == 'u-SVD':
        # u-SVD: Chỉ sử dụng U để biểu diễn dữ liệu
        all_embeddings_refined = U
    elif dimension_refinement_method == 'SVD-LR':
        # SVD-LR: Sử dụng U * Sigma, sau đó loại bỏ LDD mạnh nhất
        all_embeddings_refined = U @ Sigma  # (m x n_components)

        # Xác định LDD mạnh nhất bằng two-sample t-test
        num_en = doc_embeddings_en.shape[0]
        en_embeddings = all_embeddings_refined[:num_en, :]
        lang_embeddings = all_embeddings_refined[num_en:, :]

        # Thực hiện t-test cho từng chiều
        t_stats = []
        for dim in range(n_components):
            t_stat, _ = ttest_ind(en_embeddings[:, dim], lang_embeddings[:, dim])
            t_stats.append(abs(t_stat))
        
        # Tìm chiều có t-statistic lớn nhất (LDD mạnh nhất)
        most_influential_ldd = np.argmax(t_stats)
        print(f"Loại bỏ chiều LDD mạnh nhất: {most_influential_ldd}")

        # Loại bỏ chiều LDD mạnh nhất
        all_embeddings_refined = np.delete(all_embeddings_refined, most_influential_ldd, axis=1)
    else:
        raise ValueError("Phương pháp tinh chỉnh chiều không hợp lệ. Chọn 'u-SVD' hoặc 'SVD-LR'.")

    # Thực hiện clustering với HDBSCAN trên GPU
    print("Phân cụm với HDBSCAN trên GPU...")
    hdbscan = HDBSCAN(min_cluster_size=min_cluster_size,
                      min_samples=min_samples,
                      cluster_selection_method=cluster_selection_method,
                      metric='euclidean')
    labels = hdbscan.fit_predict(all_embeddings_refined)

    # Tách nhãn cluster
    num_en = doc_embeddings_en.shape[0]
    labels_en = labels[:num_en]
    labels_lang = labels[num_en:]

    # Tính số điểm nhiễu và số cluster
    noise_en = np.sum(labels_en == -1)
    noise_lang = np.sum(labels_lang == -1)
    total_noise = noise_en + noise_lang
    num_clusters = len(np.unique(labels[labels != -1]))

    # In kết quả
    print(f"Kết quả phân cụm cho {dataset}:")
    print(f"  - Số cluster: {num_clusters}")
    print(f"  - Số điểm nhiễu trong {dataset}:")
    print(f"    + Tiếng Anh: {noise_en} / {num_en} (chiếm {noise_en/num_en*100:.2f}%)")
    print(f"    + {lang}: {noise_lang} / {len(labels_lang)} (chiếm {noise_lang/len(labels_lang)*100:.2f}%)")
    print(f"    + Tổng cộng: {total_noise} / {len(all_embeddings)} (chiếm {total_noise/len(all_embeddings)*100:.2f}%)")

    # Lưu nhãn
    np.save(os.path.join(base_data_dir, dataset, 'cluster_labels_en_train.npy'), labels_en)
    np.save(os.path.join(base_data_dir, dataset, f'cluster_labels_{lang}_train.npy'), labels_lang)

    print(f"Đã hoàn thành clustering cho {dataset}. Nhãn đã được lưu.\n")