import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import dask.array as da
from pathlib import Path
import sys
# Import cho randomized_svd nếu cần fallback hiệu năng
# from sklearn.utils.extmath import randomized_svd

# --- Helper Functions ---

# << HÀM center_embeddings_per_language ĐÃ BỊ XÓA >>

# << HÀM LIR - ÁP DỤNG TRÊN EMBEDDING GỐC >>
def apply_lir(embeddings, lang_index, num_dims_to_remove):
    """
    Applies LIR to remove language components from the *original*
    d-dimensional embeddings (without centering).
    """
    n_samples, n_features = embeddings.shape
    # Start with a copy to avoid modifying the original array if LIR fails partially
    processed_embeddings = embeddings.copy()

    if num_dims_to_remove <= 0:
        return processed_embeddings # Return original if removing 0 or negative dims

    for lang_label in np.unique(lang_index):
        lang_mask = (lang_index == lang_label)
        lang_subset = embeddings[lang_mask] # Use original embeddings subset
        n_samples_lang = lang_subset.shape[0]

        if n_samples_lang == 0: continue

        # Cannot remove more components than samples or features
        effective_r_lir = min(num_dims_to_remove, n_samples_lang, n_features)
        if effective_r_lir < 1: continue

        try:
            # Perform SVD on the original-space language subset
            _, _, Vt = np.linalg.svd(lang_subset, full_matrices=False)
            V = Vt.T

            actual_svd_dims = V.shape[1]
            final_r_lir = min(effective_r_lir, actual_svd_dims)
            if final_r_lir < 1: continue

            c_L = V[:, :final_r_lir]

            # Create removal matrix: I - c_L @ c_L.T
            projection_matrix = c_L @ c_L.T
            identity_matrix = np.identity(n_features)
            removal_matrix = identity_matrix - projection_matrix

            # Apply removal matrix to the language subset
            processed_embeddings[lang_mask] = (removal_matrix @ lang_subset.T).T

        except np.linalg.LinAlgError:
            # Keep original subset if SVD fails
            pass
        except Exception:
            # Keep original on other unexpected errors
            pass

    return processed_embeddings
# << KẾT THÚC HÀM LIR >>


def perform_svd_reduction(embeddings, r):
    """Performs SVD and reduces dimensionality to r."""
    # Input embeddings here are expected to be LIR-processed (no centering)
    n_samples, n_features = embeddings.shape
    effective_r = min(r, n_samples, n_features)

    if effective_r < 1:
        # Fallback: scale and truncate. Note: Scaling without centering might be less standard
        scaler = StandardScaler(with_mean=False) # Scale but don't center if input wasn't centered
        try:
            embeddings_scaled = scaler.fit_transform(embeddings)
        except ValueError: # Handle cases where variance is zero
            embeddings_scaled = embeddings # Keep original if scaling fails
        safe_dim = min(embeddings_scaled.shape[1], r if r > 0 else 1)
        if safe_dim <= 0: safe_dim = 1
        if embeddings_scaled.shape[1] < safe_dim: safe_dim = embeddings_scaled.shape[1]
        truncated = embeddings_scaled[:, :safe_dim]
        return truncated, truncated

    embeddings_da = da.from_array(embeddings, chunks='auto')
    try:
        u, s, v = da.linalg.svd(embeddings_da)
        computed_r = min(effective_r, len(s))
        if computed_r < 1: raise ValueError("SVD resulted in < 1 singular value")

        u = u[:, :computed_r].compute()
        s = s[:computed_r].compute()
        return u, u * s
    except Exception as e:
        scaler = StandardScaler(with_mean=False)
        try:
            embeddings_scaled = scaler.fit_transform(embeddings)
        except ValueError:
            embeddings_scaled = embeddings
        safe_dim = min(embeddings_scaled.shape[1], r if r > 0 else 1)
        if safe_dim <= 0: safe_dim = 1
        if embeddings_scaled.shape[1] < safe_dim: safe_dim = embeddings_scaled.shape[1]
        truncated = embeddings_scaled[:, :safe_dim]
        return truncated, truncated

def compute_language_balance_std(cluster_labels, lang_index, K):
    """Calculates the standard deviation of language ratios across clusters."""
    ratios = []
    unique_labels = np.unique(cluster_labels)
    for k in unique_labels:
        cluster_indices = np.where(cluster_labels == k)[0]
        if len(cluster_indices) == 0: continue
        lang0_count = np.sum(lang_index[cluster_indices] == 0)
        total_count = len(cluster_indices)
        ratio = lang0_count / total_count if total_count > 0 else 0.5
        ratios.append(ratio)

    if not ratios: return float('inf')
    return np.std(ratios) if len(ratios) > 1 else 0.0

def scale_embeddings(embeddings):
    """Applies StandardScaler (with centering) to embeddings before KMeans."""
    # Even if input wasn't centered, scaling+centering before KMeans is standard
    if embeddings.shape[0] > 1:
        scaler = StandardScaler(with_mean=True) # Use default centering here
        try:
             return scaler.fit_transform(embeddings)
        except ValueError: # Handle zero variance case
             return embeddings # Return unscaled if scaling fails
    return embeddings

def run_kmeans(embeddings, k, n_samples):
    """Runs KMeans clustering."""
    effective_k = min(k, n_samples)
    if effective_k < 1 or embeddings.shape[1] < 1:
        print(f"Error: Cannot run KMeans (K={k}, EffectiveK={effective_k}, Samples={n_samples}, Features={embeddings.shape[1]})", flush=True)
        return None
    try:
        kmeans = KMeans(n_clusters=effective_k, random_state=0, n_init=10, max_iter=1000)
        return kmeans.fit(embeddings)
    except Exception as e:
        print(f"Error during K-Means: {e}", flush=True)
        return None

# --- Main Processing Function ---

def process_dataset(dataset_name, lang1, lang2, svd_r, lir_dims_to_remove, kmeans_k):
    """Loads, processes (LIR before SVD, no centering), clusters, evaluates, saves."""

    # 1. Load and Prepare Data
    base_path = os.environ.get("DUALTOPIC_DATA_PATH", "/content/drive/MyDrive/projects/DualTopic/data")
    embed_path_lang1 = Path(f"{base_path}/{dataset_name}/doc_embeddings_{lang1}_train.npy")
    embed_path_lang2 = Path(f"{base_path}/{dataset_name}/doc_embeddings_{lang2}_train.npy")
    save_dir = Path(f"{base_path}/{dataset_name}")

    if not embed_path_lang1.exists() or not embed_path_lang2.exists():
        print(f"File không tồn tại: {embed_path_lang1 if not embed_path_lang1.exists() else embed_path_lang2}", flush=True)
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        embed_lang1 = np.load(embed_path_lang1)
        embed_lang2 = np.load(embed_path_lang2)
    except Exception as e:
        print(f"Lỗi khi tải file embedding: {e}", flush=True)
        return

    if embed_lang1.size == 0 or embed_lang2.size == 0 or embed_lang1.shape[1] != embed_lang2.shape[1]:
        print(f"Lỗi: Embedding rỗng hoặc số chiều không khớp cho {dataset_name}", flush=True)
        return

    min_samples = min(len(embed_lang1), len(embed_lang2))
    if min_samples == 0:
        print(f"Lỗi: Không có mẫu trong một ngôn ngữ cho {dataset_name}", flush=True)
        return
    if len(embed_lang1) > min_samples:
        embed_lang1 = embed_lang1[np.random.choice(len(embed_lang1), min_samples, replace=False)]
    if len(embed_lang2) > min_samples:
        embed_lang2 = embed_lang2[np.random.choice(len(embed_lang2), min_samples, replace=False)]

    embeddings = np.concatenate([embed_lang1, embed_lang2], axis=0)
    lang_index = np.concatenate([np.zeros(min_samples), np.ones(min_samples)])
    n_total_samples = embeddings.shape[0]

    # 2. Embedding Processing Pipeline (LIR before SVD, NO Centering)
    # << BƯỚC CENTERING ĐÃ BỊ XÓA >>
    # embeddings_centered = center_embeddings_per_language(embeddings, lang_index)

    # << ÁP DỤNG LIR TRÊN EMBEDDING GỐC >>
    embeddings_lir = apply_lir(embeddings, lang_index, num_dims_to_remove=lir_dims_to_remove)

    # << THỰC HIỆN SVD SAU LIR >>
    embed_usvd, embed_svdlr = perform_svd_reduction(embeddings_lir, r=svd_r)

    if embed_usvd.size == 0 or embed_svdlr.size == 0:
         print(f"Lỗi: Kết quả SVD không hợp lệ cho {dataset_name}", flush=True)
         return

    # Path A: u-SVD Processing (derived from LIR'd data)
    embed_usvd_final = scale_embeddings(embed_usvd) # Scale+Center before KMeans

    # Path B: SVD-LR Processing (derived from LIR'd data)
    embed_svdlr_final = scale_embeddings(embed_svdlr) # Scale+Center before KMeans

    # 3. Clustering and Evaluation
    kmeans_usvd_model = run_kmeans(embed_usvd_final, kmeans_k, n_total_samples)
    kmeans_svdlr_model = run_kmeans(embed_svdlr_final, kmeans_k, n_total_samples)

    if kmeans_usvd_model is None or kmeans_svdlr_model is None:
        return

    effective_k = kmeans_usvd_model.n_clusters

    std_usvd = compute_language_balance_std(kmeans_usvd_model.labels_, lang_index, effective_k)
    std_svdlr = compute_language_balance_std(kmeans_svdlr_model.labels_, lang_index, effective_k)

    # 4. Select Best Result and Save
    if std_usvd <= std_svdlr:
        best_method_name = "u-SVD"
        best_labels = kmeans_usvd_model.labels_
        best_std = std_usvd
    else:
        best_method_name = "SVD-LR"
        best_labels = kmeans_svdlr_model.labels_
        best_std = std_svdlr

    labels_lang1 = best_labels[lang_index == 0]
    labels_lang2 = best_labels[lang_index == 1]

    save_path_lang1 = save_dir / f"cluster_labels_{lang1}.npy"
    save_path_lang2 = save_dir / f"cluster_labels_{lang2}.npy"

    try:
        np.save(save_path_lang1, labels_lang1)
        np.save(save_path_lang2, labels_lang2)
        print(f"Đã lưu nhãn cụm vào: {save_path_lang1}", flush=True)
        print(f"Đã lưu nhãn cụm vào: {save_path_lang2}", flush=True)
    except Exception as e:
        print(f"Lỗi khi lưu file nhãn cụm: {e}", flush=True)
        return

    print(f"Tập dữ liệu: {dataset_name} (Phương pháp: {best_method_name}, std={best_std:.4f})", flush=True)
    for k in range(effective_k):
        cluster_indices = np.where(best_labels == k)[0]
        if len(cluster_indices) == 0: continue
        lang0_count = np.sum(lang_index[cluster_indices] == 0)
        lang1_count = np.sum(lang_index[cluster_indices] == 1)
        print(f"Cụm {k}: {lang0_count} văn bản từ {lang1}, {lang1_count} văn bản từ {lang2}", flush=True)

# --- Main Execution ---
if __name__ == "__main__":
    print("Kết quả phân cụm:\n", flush=True)

    SVD_DIM = 50
    LIR_DIMS_TO_REMOVE = 20
    KMEANS_CLUSTERS = 20

    datasets_to_process = [
        ('Amazon_Review', 'cn', 'en'),
        ('ECNews', 'cn', 'en'),
        ('Rakuten_Amazon', 'ja', 'en')
    ]

    for i, (dataset, l1, l2) in enumerate(datasets_to_process):
        process_dataset(dataset_name=dataset,
                        lang1=l1,
                        lang2=l2,
                        svd_r=SVD_DIM,
                        lir_dims_to_remove=LIR_DIMS_TO_REMOVE,
                        kmeans_k=KMEANS_CLUSTERS)
        if i < len(datasets_to_process) - 1:
            print("\n", flush=True)
        sys.stdout.flush()