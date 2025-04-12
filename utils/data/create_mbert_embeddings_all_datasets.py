# create_mbert_embeddings_all_datasets.py
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import os
from tqdm import tqdm

# Hàm đọc file văn bản
def read_texts(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []

# Hàm lưu embeddings
def save_embeddings(embeddings, file_path):
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, embeddings)
    print(f"Đã lưu embeddings vào: {file_path}")

# Hàm tải embeddings (nếu đã tồn tại)
def load_embeddings(file_path):
    try:
        embeddings = np.load(file_path)
        print(f"Đã tải embeddings từ: {file_path}")
        return embeddings
    except FileNotFoundError:
        print(f"Không tìm thấy file: {file_path}")
        return None

# Hàm tạo word embeddings từ mBERT (chỉ gọi khi cần tạo mới)
def generate_word_embeddings(vocab, lang, tokenizer, bert_model, device, batch_size=16):
    if not vocab:
        print(f"Không có từ vựng nào để tạo embeddings cho {lang}. Bỏ qua...")
        return None

    print(f"Tạo word embeddings từ mBERT cho {lang}...")
    embeddings = []
    for i in tqdm(range(0, len(vocab), batch_size), desc=f"Processing word embeddings for {lang}"):
        batch_words = vocab[i:i + batch_size]
        inputs = tokenizer(
            batch_words,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = bert_model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

# Hàm tạo document embeddings từ mBERT (chỉ gọi khi cần tạo mới)
def generate_document_embeddings(texts, lang, split, tokenizer, bert_model, device, batch_size=16):
    if not texts:
        print(f"Không có văn bản nào để tạo embeddings cho {lang} ({split}). Bỏ qua...")
        return None

    print(f"Tạo document embeddings từ mBERT cho {lang} ({split})...")
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Processing document embeddings for {lang} {split}"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = bert_model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

# Hàm xử lý word embeddings (load hoặc tạo mới)
def create_word_embeddings(vocab, lang, data_dir, output_dir, tokenizer, bert_model, device, batch_size=16):
    embedding_file = os.path.join(output_dir, f'word_embeddings_{lang}.npy')
    cached_embeddings = load_embeddings(embedding_file)
    if cached_embeddings is not None:
        return cached_embeddings

    embeddings = generate_word_embeddings(vocab, lang, tokenizer, bert_model, device, batch_size)
    if embeddings is not None:
        save_embeddings(embeddings, embedding_file)
    return embeddings

# Hàm xử lý document embeddings (load hoặc tạo mới)
def create_document_embeddings(texts, lang, data_dir, output_dir, tokenizer, bert_model, device, batch_size=16, is_train=True):
    split = 'train' if is_train else 'test'
    embedding_file = os.path.join(output_dir, f'doc_embeddings_{lang}_{split}.npy')
    cached_embeddings = load_embeddings(embedding_file)
    if cached_embeddings is not None:
        return cached_embeddings

    embeddings = generate_document_embeddings(texts, lang, split, tokenizer, bert_model, device, batch_size)
    if embeddings is not None:
        save_embeddings(embeddings, embedding_file)
    return embeddings

# Hàm chính để tạo embeddings cho tất cả các dataset
def create_embeddings_for_all_datasets(base_data_dir, output_base_dir, datasets, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    if device.type == "cuda":
        print(f"Tên GPU: {torch.cuda.get_device_name(0)}")

    print("Tải mBERT (bert-base-multilingual-cased)...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    bert_model = AutoModel.from_pretrained("bert-base-multilingual-cased")
    bert_model = bert_model.to(device)
    print(f"Thiết bị của mô hình: {next(bert_model.parameters()).device}")

    # Xác định cặp ngôn ngữ cho từng dataset
    dataset_languages = {
        'Amazon_Review': ('en', 'cn'),
        'ECNews': ('en', 'cn'),
        'Rakuten_Amazon': ('en', 'ja')
    }

    for dataset_name in datasets:
        data_dir = os.path.join(base_data_dir, dataset_name)
        output_dir = os.path.join(output_base_dir, dataset_name)
        print(f"\n=== Xử lý dataset: {dataset_name} ===")
        print(f"Đang đọc dữ liệu từ: {data_dir}")
        print(f"Embeddings sẽ được lưu tại: {output_dir}")

        if not os.path.exists(data_dir):
            print(f"Thư mục {data_dir} không tồn tại. Bỏ qua dataset {dataset_name}.")
            continue

        # Lấy cặp ngôn ngữ cho dataset hiện tại
        lang1, lang2 = dataset_languages[dataset_name]
        print(f"Ngôn ngữ: {lang1}-{lang2}")

        # Đọc dữ liệu
        print(f"Reading {lang1} data...")
        train_texts_lang1 = read_texts(os.path.join(data_dir, f'train_texts_{lang1}.txt'))
        test_texts_lang1 = read_texts(os.path.join(data_dir, f'test_texts_{lang1}.txt'))
        vocab_lang1 = read_texts(os.path.join(data_dir, f'vocab_{lang1}'))

        print(f"Reading {lang2} data...")
        train_texts_lang2 = read_texts(os.path.join(data_dir, f'train_texts_{lang2}.txt'))
        test_texts_lang2 = read_texts(os.path.join(data_dir, f'test_texts_{lang2}.txt'))
        vocab_lang2 = read_texts(os.path.join(data_dir, f'vocab_{lang2}'))

        # Tạo word embeddings
        word_embeddings_lang1 = create_word_embeddings(vocab_lang1, lang1, data_dir, output_dir, tokenizer, bert_model, device, batch_size)
        word_embeddings_lang2 = create_word_embeddings(vocab_lang2, lang2, data_dir, output_dir, tokenizer, bert_model, device, batch_size)

        # Tạo document embeddings
        doc_embeddings_lang1_train = create_document_embeddings(train_texts_lang1, lang1, data_dir, output_dir, tokenizer, bert_model, device, batch_size, is_train=True)
        doc_embeddings_lang1_test = create_document_embeddings(test_texts_lang1, lang1, data_dir, output_dir, tokenizer, bert_model, device, batch_size, is_train=False)
        doc_embeddings_lang2_train = create_document_embeddings(train_texts_lang2, lang2, data_dir, output_dir, tokenizer, bert_model, device, batch_size, is_train=True)
        doc_embeddings_lang2_test = create_document_embeddings(test_texts_lang2, lang2, data_dir, output_dir, tokenizer, bert_model, device, batch_size, is_train=False)

        print(f"Hoàn tất tạo embeddings cho dataset {dataset_name}.")

if __name__ == "__main__":
    base_data_dir = '/content/drive/MyDrive/projects/InfoCTM/data'
    output_base_dir = '/content/drive/MyDrive/projects/InfoCTM/data'
    datasets = ['Amazon_Review', 'ECNews', 'Rakuten_Amazon']
    batch_size = 16

    create_embeddings_for_all_datasets(base_data_dir, output_base_dir, datasets, batch_size=batch_size)