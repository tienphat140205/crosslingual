import os
import yaml
import scipy.io
import argparse
import numpy as np

from runners.Runner import Runner
from utils.data import file_utils
from utils.data.TextData import DatasetHandler
from utils.eval import compute_topic_coherence, compute_topic_diversity, TU_eva

def export_beta(beta, vocab, output_prefix, lang, num_top_word=15):
    topic_str_list = file_utils.print_topic_words(beta, vocab, num_top_word=num_top_word)
    output_path = f'{output_prefix}_T{num_top_word}_{lang}.txt'
    file_utils.save_text(topic_str_list, path=output_path)
    print(f"Đã xuất từ chủ đề cho {lang} vào {output_path}")
    return topic_str_list

def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình DualTopic")
    parser.add_argument('--model', type=str, required=True, help="Tên mô hình")
    parser.add_argument('--dataset', type=str, required=True, help="Tên tập dữ liệu")
    parser.add_argument('--num_topic', type=int, help="Số lượng chủ đề")
    parser.add_argument('--learning_rate', type=float, help="Tốc độ học")
    parser.add_argument('--epochs', type=int, help="Số epoch huấn luyện")
    parser.add_argument('--batch_size', type=int, help="Kích thước batch")
    parser.add_argument('--lambda_contrast', type=float, help="Hệ số lambda contrast")
    parser.add_argument('--weight_MI', type=float, help="Hệ số weight cho Mutual Information loss")
    parser.add_argument('--trans_dict_path', type=str, help="Đường dẫn đến từ điển dịch")  # Xóa default
    parser.add_argument('--temperature', type=float, default=0.07, help="Nhiệt độ cho contrastive loss")
    parser.add_argument('--pos_threshold', type=float, default=0.7, help="Ngưỡng xác suất tích cực")
    return parser.parse_args()

def main():
    print("Bắt đầu chương trình...")

    # 1. Phân tích Đối số và Tải Cấu hình
    args = parse_args()
    args = file_utils.update_args(args, f'./configs/model/{args.model}.yaml')
    args = file_utils.update_args(args, f'./configs/dataset/{args.dataset}.yaml')
    cmd_args = vars(parse_args())
    for k, v in cmd_args.items():
        if v is not None and hasattr(args, k):
            setattr(args, k, v)

    # 2. Thiết lập Đường dẫn Đầu ra
    output_dir = os.path.abspath(f'output/{args.dataset}')
    run_id = f'{args.model}_K{args.num_topic}_LR{args.learning_rate}_LC{args.lambda_contrast}_WMI{args.weight_MI}_E{args.epochs}'    
    output_prefix = os.path.join(output_dir, run_id)
    file_utils.make_dir(output_dir)
    args.output_dir = output_dir
    args.output_prefix = output_prefix

    print("Cấu hình chương trình:")
    print(yaml.dump(vars(args), default_flow_style=False, indent=2))
    with open(f"{output_prefix}_config.yaml", 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    # 3. Tải Dữ liệu
    data_root = f'./data/{args.dataset}'
    print(f"Tải dữ liệu từ: {data_root}")
    dataset_handler = DatasetHandler(args.dataset, args.batch_size, args.lang1, args.lang2)
    vocab_path_lang1 = os.path.join(data_root, f'vocab_{args.lang1}')
    vocab_path_lang2 = os.path.join(data_root, f'vocab_{args.lang2}')
    embed_path_lang1 = os.path.join(data_root, f'word_embeddings_{args.lang1}.npy')
    embed_path_lang2 = os.path.join(data_root, f'word_embeddings_{args.lang2}.npy')

    vocab_lang1 = file_utils.read_texts(vocab_path_lang1) if os.path.exists(vocab_path_lang1) else []
    vocab_lang2 = file_utils.read_texts(vocab_path_lang2) if os.path.exists(vocab_path_lang2) else []
    vocab_lang1_dict = {word: idx for idx, word in enumerate(vocab_lang1)}
    vocab_lang2_dict = {word: idx for idx, word in enumerate(vocab_lang2)}
    vocab_size_lang1 = len(vocab_lang1)
    vocab_size_lang2 = len(vocab_lang2)

    word_embeddings_lang1 = np.load(embed_path_lang1)
    word_embeddings_lang2 = np.load(embed_path_lang2)

    with open(os.path.join(data_root, f'train_texts_{args.lang1}.txt'), 'r', encoding='utf-8') as f:
        documents_lang1 = f.readlines()
    with open(os.path.join(data_root, f'train_texts_{args.lang2}.txt'), 'r', encoding='utf-8') as f:
        documents_lang2 = f.readlines()

    print(f"Kích thước từ vựng {args.lang1}: {vocab_size_lang1}, {args.lang2}: {vocab_size_lang2}")
    print(f"Kích thước dữ liệu huấn luyện: {dataset_handler.train_size}")

    # 4. Chuẩn bị Tham số cho Mô hình
    model_params_list = [
        vocab_size_lang1, vocab_size_lang2, args.num_topic, args.hidden_dim,
        word_embeddings_lang1, word_embeddings_lang2, vocab_lang1_dict, vocab_lang2_dict,
        args.trans_dict_path, args.dropout, args.tau, args.lambda_contrast, args.weight_MI, 
        args.temperature, args.pos_threshold
    ]

    # 5. Khởi tạo Runner
    print("Khởi tạo Runner...")
    runner = Runner(args, model_params_list, args.lang1, args.lang2)
    runner.log_file_path = f'{output_prefix}_epoch_log.jsonl'
    
    # 6. Huấn luyện Mô hình
    print("Bắt đầu huấn luyện...")
    train_results = runner.train(dataset_handler.train_loader)
    beta_lang1 = train_results.get('beta_lang1')
    beta_lang2 = train_results.get('beta_lang2')

    # 7. Xuất Beta (Từ Chủ đề)
    print("Xuất từ chủ đề (beta)...")
    topic_str_list_lang1 = export_beta(beta_lang1, vocab_lang1, output_prefix, args.lang1)
    topic_str_list_lang2 = export_beta(beta_lang2, vocab_lang2, output_prefix, args.lang2)

    # 8. Đánh giá Mô hình
    print("Đánh giá trên tập huấn luyện và kiểm tra...")
    train_theta_lang1, train_theta_lang2 = runner.test(dataset_handler.train_loader)
    test_theta_lang1, test_theta_lang2 = runner.test(dataset_handler.test_loader)

    # 9. Đánh giá Chất lượng Chủ đề
    print("Tính toán các chỉ số chất lượng chủ đề...")
    tu_lang1 = TU_eva(topic_str_list_lang1)
    tu_lang2 = TU_eva(topic_str_list_lang2)
    td_lang1 = compute_topic_diversity(topic_str_list_lang1, _type="TD")
    td_lang2 = compute_topic_diversity(topic_str_list_lang2, _type="TD")
    tc_lang1_per_topic, tc_lang1 = compute_topic_coherence(documents_lang1, vocab_lang1, topic_str_list_lang1, cv_type='c_v')
    tc_lang2_per_topic, tc_lang2 = compute_topic_coherence(documents_lang2, vocab_lang2, topic_str_list_lang2, cv_type='c_v')

    print(f"\nChỉ số Chất lượng Chủ đề cho {args.lang1}:")
    print(f"  TU: {tu_lang1:.4f}")
    print(f"  TD: {td_lang1:.4f}")
    print(f"  TC (C_V): {tc_lang1:.4f}")
    print(f"\nChỉ số Chất lượng Chủ đề cho {args.lang2}:")
    print(f"  TU: {tu_lang2:.4f}")
    print(f"  TD: {td_lang2:.4f}")
    print(f"  TC (C_V): {tc_lang2:.4f}")

    # 10. Lưu Kết quả
    print("Lưu kết quả vào tệp .mat...")
    rst_dict = {
        f'beta_{args.lang1}': beta_lang1,
        f'beta_{args.lang2}': beta_lang2,
        f'train_theta_{args.lang1}': train_theta_lang1,
        f'train_theta_{args.lang2}': train_theta_lang2,
        f'test_theta_{args.lang1}': test_theta_lang1,
        f'test_theta_{args.lang2}': test_theta_lang2,
        'config': vars(args),
        'tu_lang1': tu_lang1,
        'tu_lang2': tu_lang2,
        'td_lang1': td_lang1,
        'td_lang2': td_lang2,
        'tc_lang1': tc_lang1,
        'tc_lang2': tc_lang2
    }
    scipy.io.savemat(f'{output_prefix}_results.mat', rst_dict, do_compression=True)
    print(f"Kết quả đã được lưu vào {output_prefix}_results.mat")

    # 11. In Các Chủ đề Mẫu
    print("\nCác Chủ đề Mẫu:")
    for i in range(min(5, args.num_topic)):
        print(f"--- Chủ đề {i} ---")
        print(f"  {args.lang1.upper()}: {topic_str_list_lang1[i]}")
        print(f"  {args.lang2.upper()}: {topic_str_list_lang2[i]}")

if __name__ == '__main__':
    main()