import numpy as np
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import dask.array as da
from pathlib import Path

# Hàm căn giữa và chuẩn hóa embeddings cho từng ngôn ngữ
def embed_centering(embeddings, lang_index):
    embeddings_1 = embeddings[lang_index == 0]  # Ngôn ngữ 1
    embeddings_2 = embeddings[lang_index == 1]  # Ngôn ngữ 2
    scaler = StandardScaler()
    embeddings_1_center = scaler.fit_transform(embeddings_1)
    embeddings_2_center = scaler.fit_transform(embeddings_2)
    return np.concatenate((embeddings_1_center, embeddings_2_center), axis=0)

# Hàm thực hiện SVD
def embed_SVD(embeddings, r=50):
    embeddings = da.from_array(embeddings)  # Dùng dask để xử lý dữ liệu lớn
    u, s, v = da.linalg.svd(embeddings)
    u = u[:, :r].compute()  # Giảm xuống r chiều
    s = s[:r].compute()
    return u, u * s  # Trả về U (u-SVD) và U * Sigma (SVD-LR)

# Hàm loại bỏ các chiều ngôn ngữ trong SVD-LR
def lang_dim_remove(embed_scale, lang_index, num_dims_to_remove=3):
    embed_scale_lang1 = embed_scale[lang_index == 0]
    embed_scale_lang2 = embed_scale[lang_index == 1]
    def t_test(dim):
        res = ttest_ind(embed_scale_lang1[:, dim], embed_scale_lang2[:, dim])
        return res.statistic, res.pvalue
    res = [t_test(dim) for dim in range(embed_scale.shape[1])]
    ttest_df = pd.DataFrame(res, columns=["statistic", "pvalue"])
    ttest_df["statistic"] = np.abs(ttest_df["statistic"])
    top_dims = ttest_df["statistic"].nlargest(num_dims_to_remove).index
    return np.delete(embed_scale, top_dims, axis=1)

# Hàm tính độ lệch chuẩn của tỷ lệ ngôn ngữ trong các cụm
def compute_language_balance(cluster_labels, lang_index, K):
    ratios = []
    for k in range(K):
        cluster_indices = np.where(cluster_labels == k)[0]
        if len(cluster_indices) == 0:
            continue
        lang0_count = np.sum(lang_index[cluster_indices] == 0)
        total_count = len(cluster_indices)
        ratio = lang0_count / total_count if total_count > 0 else 0.5
        ratios.append(ratio)
    return np.std(ratios) if ratios else float('inf')

# Hàm xử lý từng tập dữ liệu và lưu nhãn cụm
def process_dataset(dataset_name, lang1, lang2):
    base_path = "/content/drive/MyDrive/projects/DualTopic/data"
    embed_path_lang1 = f"{base_path}/{dataset_name}/doc_embeddings_{lang1}_train.npy"
    embed_path_lang2 = f"{base_path}/{dataset_name}/doc_embeddings_{lang2}_train.npy"
    save_dir = f"{base_path}/{dataset_name}"
    
    # Kiểm tra file tồn tại
    for path in [embed_path_lang1, embed_path_lang2]:
        if not os.path.exists(path):
            print(f"File không tồn tại: {path}")
            return

    # Tạo thư mục lưu nếu chưa có
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Tải embeddings
    embed_lang1 = np.load(embed_path_lang1)
    embed_lang2 = np.load(embed_path_lang2)
    
    # Cân bằng số văn bản giữa hai ngôn ngữ
    min_samples = min(len(embed_lang1), len(embed_lang2))
    if len(embed_lang1) > min_samples:
        indices = np.random.choice(len(embed_lang1), min_samples, replace=False)
        embed_lang1 = embed_lang1[indices]
    if len(embed_lang2) > min_samples:
        indices = np.random.choice(len(embed_lang2), min_samples, replace=False)
        embed_lang2 = embed_lang2[indices]
    
    embeddings = np.concatenate([embed_lang1, embed_lang2], axis=0)
    lang_index = np.concatenate([np.zeros(len(embed_lang1)), np.ones(len(embed_lang2))])
    
    # Pipeline xử lý embeddings
    embeddings_center = embed_centering(embeddings, lang_index)
    embed_usvd, embed_svdlr = embed_SVD(embeddings_center, r=50)
    
    # u-SVD: Chuẩn hóa sau SVD
    scaler = StandardScaler()
    embed_usvd = scaler.fit_transform(embed_usvd)
    
    # SVD-LR: Loại bỏ chiều ngôn ngữ và chuẩn hóa
    embed_svdlr = lang_dim_remove(embed_svdlr, lang_index, num_dims_to_remove=3)
    embed_svdlr = scaler.fit_transform(embed_svdlr)
    
    # Phân cụm với K-means (K=20)
    K = 20
    kmeans_usvd = KMeans(n_clusters=K, random_state=0).fit(embed_usvd)
    kmeans_svdlr = KMeans(n_clusters=K, random_state=0).fit(embed_svdlr)
    
    # Đánh giá độ cân bằng ngôn ngữ
    std_usvd = compute_language_balance(kmeans_usvd.labels_, lang_index, K)
    std_svdlr = compute_language_balance(kmeans_svdlr.labels_, lang_index, K)
    
    # Chọn phương pháp tốt hơn
    best_method = "u-SVD" if std_usvd < std_svdlr else "SVD-LR"
    best_labels = kmeans_usvd.labels_ if std_usvd < std_svdlr else kmeans_svdlr.labels_
    
    # Tách nhãn cụm theo ngôn ngữ
    labels_lang1 = best_labels[lang_index == 0]  # Nhãn cho ngôn ngữ 1
    labels_lang2 = best_labels[lang_index == 1]  # Nhãn cho ngôn ngữ 2
    
    # Lưu nhãn cụm vào hai file riêng
    save_path_lang1 = f"{save_dir}/cluster_labels_{lang1}.npy"
    save_path_lang2 = f"{save_dir}/cluster_labels_{lang2}.npy"
    np.save(save_path_lang1, labels_lang1)
    np.save(save_path_lang2, labels_lang2)
    print(f"Đã lưu nhãn cụm vào: {save_path_lang1}")
    print(f"Đã lưu nhãn cụm vào: {save_path_lang2}")
    
    # In kết quả
    print(f"Tập dữ liệu: {dataset_name} (Phương pháp: {best_method}, std={min(std_usvd, std_svdlr):.4f})")
    for k in range(K):
        cluster_indices = np.where(best_labels == k)[0]
        lang0_count = np.sum(lang_index[cluster_indices] == 0)
        lang1_count = np.sum(lang_index[cluster_indices] == 1)
        print(f"Cụm {k}: {lang0_count} văn bản từ {lang1}, {lang1_count} văn bản từ {lang2}")

# Chạy mã cho các tập dữ liệu
if __name__ == "__main__":
    print("Kết quả phân cụm:\n")
    process_dataset('Amazon_Review', 'cn', 'en')
    print("\n")
    process_dataset('ECNews', 'cn', 'en')
    print("\n")
    process_dataset('Rakuten_Amazon', 'ja', 'en')