import os
import yaml
import scipy.io
from runners.Runner import Runner
import argparse

from utils.data import file_utils
from utils.data.TextData import DatasetHandler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--dataset')
    parser.add_argument('--weight_MI', type=float)
    parser.add_argument('--num_topic', type=int, default=50)
    parser.add_argument('--lambda_contrast', type=float)
    parser.add_argument('--infoncealpha', type=float, default=1.0, help='Weight for InfoNCE loss')

    parser.add_argument('--device', type=int, default=0, help='CUDA device index to use')

    args = parser.parse_args()
    return args


def export_beta(beta, vocab, output_prefix, lang):
    num_top_word = 15
    topic_str_list = file_utils.print_topic_words(beta, vocab, num_top_word=num_top_word)
    file_utils.save_text(topic_str_list, path=f'{output_prefix}_T{num_top_word}_{lang}.txt')
    return topic_str_list


def main():
    args = parse_args()

    args = file_utils.update_args(args, f'./configs/model/{args.model}.yaml')

    args = file_utils.update_args(args, f'./configs/dataset/{args.dataset}.yaml')

    output_prefix = f'output/{args.dataset}/{args.model}_K{args.num_topic}'
    file_utils.make_dir(os.path.dirname(output_prefix))

    print('\n' + yaml.dump(vars(args), default_flow_style=False))

    dataset_handler = DatasetHandler(args.dataset, args.batch_size, args.lang1, args.lang2, args.dict_path, device=args.device)

    args.cluster_en=dataset_handler.clusterinfo_en
    args.cluster_cn=dataset_handler.clusterinfo_cn
    args.vocab_size_en = len(dataset_handler.vocab_en)
    args.vocab_size_cn = len(dataset_handler.vocab_cn)

    args.mask_en_to_cn = dataset_handler.mask_en_to_cn
    args.mask_cn_to_en = dataset_handler.mask_cn_to_en

    args.pretrain_word_embeddings_en = dataset_handler.pretrain_word_embeddings_en
    args.pretrain_word_embeddings_cn = dataset_handler.pretrain_word_embeddings_cn

    args.vocab_en = dataset_handler.vocab_en
    args.vocab_cn = dataset_handler.vocab_cn

    params_list = [dataset_handler.trans_matrix_en]


    

    runner = Runner(args, params_list)

    beta_en, beta_cn = runner.train(dataset_handler.train_loader)

    topic_str_list_en = export_beta(beta_en, dataset_handler.vocab_en, output_prefix, lang='en')
    topic_str_list_cn = export_beta(beta_cn, dataset_handler.vocab_cn, output_prefix, lang='cn')

    for i in range(len(topic_str_list_en)):
        print(topic_str_list_en[i])
        print(topic_str_list_cn[i])

    train_theta_en, train_theta_cn = runner.test(dataset_handler.train_loader.dataset)
    test_theta_en, test_theta_cn = runner.test(dataset_handler.test_loader.dataset)

    rst_dict = {
        'beta_en': beta_en,
        'beta_cn': beta_cn,
        'train_theta_en': train_theta_en,
        'train_theta_cn': train_theta_cn,
        'test_theta_en': test_theta_en,
        'test_theta_cn': test_theta_cn,
    }

    scipy.io.savemat(f'{output_prefix}_rst.mat', rst_dict)


if __name__ == '__main__':
    main()
