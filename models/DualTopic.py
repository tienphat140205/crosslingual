import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.networks.Encoder import MLPEncoder

class DualTopic(nn.Module):
    def __init__(self, vocab_size_lang1, vocab_size_lang2, num_topics, hidden_dim, 
                 word_embeddings_lang1, word_embeddings_lang2, vocab_lang1_dict, vocab_lang2_dict, 
                 trans_dict_path, dropout=0.3, tau=1.0, lambda_contrast=5, weight_MI=1.0, 
                 temperature=0.07, pos_threshold=0.7):
        super(DualTopic, self).__init__()
        self.num_topics = num_topics
        self.tau = tau
        self.lambda_contrast = lambda_contrast
        self.weight_MI = weight_MI
        self.temperature = temperature
        self.pos_threshold = pos_threshold

        # Encoders cho từng ngôn ngữ
        self.encoder_lang1 = MLPEncoder(vocab_size_lang1, num_topics, hidden_dim, dropout)
        self.encoder_lang2 = MLPEncoder(vocab_size_lang2, num_topics, hidden_dim, dropout)

        # Word embeddings (đóng băng - không huấn luyện)
        self.word_embeddings_lang1 = nn.Parameter(torch.from_numpy(word_embeddings_lang1).float(), requires_grad=False)
        self.word_embeddings_lang2 = nn.Parameter(torch.from_numpy(word_embeddings_lang2).float(), requires_grad=False)

        # Topic embeddings (có thể huấn luyện)
        self.topic_embeddings_lang1 = nn.Parameter(torch.randn(num_topics, self.word_embeddings_lang1.shape[1]) * 0.01)
        self.topic_embeddings_lang2 = nn.Parameter(torch.randn(num_topics, self.word_embeddings_lang2.shape[1]) * 0.01)

        # Tham số prior
        self.a = 1 * np.ones((1, num_topics)).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T), requires_grad=False)
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T), requires_grad=False)

        # Batch normalization cho decoder
        self.decoder_bn_lang1 = nn.BatchNorm1d(vocab_size_lang1, affine=True)
        self.decoder_bn_lang1.weight.requires_grad = False
        self.decoder_bn_lang2 = nn.BatchNorm1d(vocab_size_lang2, affine=True)
        self.decoder_bn_lang2.weight.requires_grad = False

        # Lưu trữ từ điển từ vựng
        self.vocab_lang1_dict = vocab_lang1_dict  # {word: index}
        self.vocab_lang2_dict = vocab_lang2_dict  # {word: index}

        # Khởi tạo masks cho Mutual Information loss
        self.init_masks(trans_dict_path)

    def decode(self, theta, beta, lang):
        """Giải mã để tái tạo đầu vào."""
        bn = getattr(self, f'decoder_bn_{lang}')
        recon = F.softmax(bn(torch.matmul(theta, beta)), dim=1)
        return recon

    def tm_loss(self, x_bow, theta, beta, mu, logvar, lang):
        """Tính toán mất mát của mô hình chủ đề."""
        recon = self.decode(theta, beta, lang)
        recon_loss = -(x_bow * (recon + 1e-10).log()).sum(1).mean()
        
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        kl_loss = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.num_topics).mean()
        
        return recon_loss + kl_loss

    def compute_beta(self, word_embeddings, topic_embeddings):
        """Tính toán phân phối từ trên các chủ đề (beta)."""
        dist = torch.cdist(word_embeddings.unsqueeze(0), topic_embeddings.unsqueeze(0), p=2).squeeze(0)
        beta = torch.exp(-dist / self.tau)
        beta = beta / beta.sum(dim=0, keepdim=True)
        return beta.transpose(0, 1)

    def contrastive_loss(self, theta_lang1, theta_lang2, cluster_info, batch_size):
        """Tính toán mất mát tương phản."""
        theta_all = torch.cat([theta_lang1, theta_lang2], dim=0)
        cluster_all = [c[0] for c in cluster_info] + [c[1] for c in cluster_info]
        
        theta_norm = F.normalize(theta_all, dim=-1)
        sim_matrix = torch.matmul(theta_norm, theta_norm.T) / self.tau
        sim_exp = torch.exp(sim_matrix)
        
        eye_mask = torch.eye(2 * batch_size, device=theta_all.device, dtype=torch.bool)
        sim_exp = sim_exp * (~eye_mask).float()
        
        cluster_all_tensor = torch.tensor(cluster_all, device=theta_all.device)
        pos_mask = (cluster_all_tensor.unsqueeze(0) == cluster_all_tensor.unsqueeze(1)).float()
        pos_mask = pos_mask * (~eye_mask).float()
        
        pos_sim = torch.sum(sim_exp * pos_mask, dim=1)
        neg_sum = torch.sum(sim_exp, dim=1) - sim_exp.diag()
        
        loss_per_anchor = -torch.log(pos_sim / (pos_sim + neg_sum + 1e-8) + 1e-8)
        valid_anchors = (pos_sim > 0).float()
        count = torch.sum(valid_anchors)
        total_loss = torch.sum(loss_per_anchor * valid_anchors)
        return (total_loss / (count + 1e-8)) * self.lambda_contrast 

    def pos_neg_mono_mask(self, embeddings, _type='pos'):
        norm_embed = F.normalize(embeddings)
        cos_sim = torch.matmul(norm_embed, norm_embed.T)
        if _type == 'pos':
            return (cos_sim >= self.pos_threshold).float()

    def translation_mask(self, mask, trans_dict_matrix):
        """Tạo mask dịch dựa trên từ điển dịch."""
        return torch.matmul(mask, trans_dict_matrix)

    def init_masks(self, trans_dict_path):
        """Khởi tạo các mask cho mất mát Mutual Information."""
        trans_lang1_to_lang2 = self.load_trans_dict(trans_dict_path)
        self.trans_lang1_to_lang2 = nn.Parameter(trans_lang1_to_lang2, requires_grad=False)
        self.trans_lang2_to_lang1 = nn.Parameter(self.trans_lang1_to_lang2.T, requires_grad=False)

        pos_mono_mask_lang1 = self.pos_neg_mono_mask(self.word_embeddings_lang1, _type='pos')
        pos_mono_mask_lang2 = self.pos_neg_mono_mask(self.word_embeddings_lang2, _type='pos')

        pos_trans_mask_lang1 = self.translation_mask(pos_mono_mask_lang1, self.trans_lang1_to_lang2)
        pos_trans_mask_lang2 = self.translation_mask(pos_mono_mask_lang2, self.trans_lang2_to_lang1)

        neg_trans_mask_lang1 = (pos_trans_mask_lang1 <= 0).float()
        neg_trans_mask_lang2 = (pos_trans_mask_lang2 <= 0).float()

        self.pos_trans_mask_lang1 = nn.Parameter(pos_trans_mask_lang1, requires_grad=False)
        self.pos_trans_mask_lang2 = nn.Parameter(pos_trans_mask_lang2, requires_grad=False)
        self.neg_trans_mask_lang1 = nn.Parameter(neg_trans_mask_lang1, requires_grad=False)
        self.neg_trans_mask_lang2 = nn.Parameter(neg_trans_mask_lang2, requires_grad=False)


    # Giả sử hàm này nằm trong một class như trong code gốc của bạn
    def load_trans_dict(self, trans_dict_path):
        """Tải từ điển dịch từ tệp và tạo ma trận dịch."""
        # Giả định:
        # lang1 là tiếng Anh (en)
        # lang2 là tiếng Trung (cn)
        # self.vocab_lang1_dict là từ điển map từ tiếng Anh -> index
        # self.vocab_lang2_dict là từ điển map từ tiếng Trung -> index
        # self.word_embeddings_lang1.shape[0] là kích thước từ vựng tiếng Anh (V1)
        # self.word_embeddings_lang2.shape[0] là kích thước từ vựng tiếng Trung (V2)
        V1, V2 = self.word_embeddings_lang1.shape[0], self.word_embeddings_lang2.shape[0]

        # Ma trận dịch: hàng là từ tiếng Anh (lang1), cột là từ tiếng Trung (lang2)
        # trans_matrix[idx_lang1, idx_lang2] = 1 nếu từ tại idx_lang1 (Anh) và idx_lang2 (Trung) là cặp dịch
        trans_matrix = torch.zeros(V1, V2)
        valid_pairs_count = 0 # Đếm số cặp hợp lệ tìm thấy

        try:
            with open(trans_dict_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    # Sử dụng split() để tách theo khoảng trắng (phổ biến hơn cho các tệp từ điển đơn giản)
                    # thay vì split('\t')
                    parts = line.strip().split()

                    # Kiểm tra xem có đúng 2 phần tử sau khi tách không
                    if len(parts) == 2:
                        # Giả định định dạng tệp là: Chinese_word English_word
                        # Dựa trên mã mẫu thứ 2 (cn_term = terms[0], en_term = terms[1])
                        # Vậy parts[0] là tiếng Trung (lang2), parts[1] là tiếng Anh (lang1)
                        word_lang2, word_lang1 = parts[0], parts[1]

                        # Kiểm tra xem cả hai từ có trong từ vựng tương ứng không
                        if word_lang1 in self.vocab_lang1_dict and word_lang2 in self.vocab_lang2_dict:
                            idx_lang1 = self.vocab_lang1_dict[word_lang1] # Index từ tiếng Anh
                            idx_lang2 = self.vocab_lang2_dict[word_lang2] # Index từ tiếng Trung

                            # Đánh dấu 1 vào vị trí tương ứng trong ma trận dịch
                            trans_matrix[idx_lang1, idx_lang2] = 1
                            valid_pairs_count += 1
                        # else:
                            # (Tùy chọn) Bạn có thể thêm log ở đây để xem từ nào không có trong từ vựng
                            # if word_lang1 not in self.vocab_lang1_dict:
                            #     print(f"Từ '{word_lang1}' (dòng {line_num+1}) không có trong vocab_lang1_dict")
                            # if word_lang2 not in self.vocab_lang2_dict:
                            #     print(f"Từ '{word_lang2}' (dòng {line_num+1}) không có trong vocab_lang2_dict")

                    # else:
                        # (Tùy chọn) Log các dòng không đúng định dạng
                        # if line.strip(): # Bỏ qua các dòng trống
                        #    print(f"Cảnh báo: Dòng {line_num+1} không đúng định dạng (kỳ vọng 2 từ): '{line.strip()}'")


            # Kiểm tra xem có tìm thấy cặp hợp lệ nào không
            # Sử dụng valid_pairs_count hoặc trans_matrix.sum() đều được ở đây vì giá trị là 0 hoặc 1
            if valid_pairs_count == 0:
                print(f"Cảnh báo: Từ điển dịch tại {trans_dict_path} không chứa cặp từ hợp lệ nào khớp với từ vựng.")
                print("Hãy kiểm tra lại:")
                print(f"1. Định dạng tệp '{trans_dict_path}' (phải là 'từ_tiếng_Trung từ_tiếng_Anh' trên mỗi dòng, phân tách bằng khoảng trắng).")
                print(f"2. Nội dung của từ điển có khớp với các từ trong 'vocab_lang1_dict' (Anh) và 'vocab_lang2_dict' (Trung) không.")
                print(f"3. Đảm bảo self.vocab_lang1_dict chứa từ vựng tiếng Anh và self.vocab_lang2_dict chứa từ vựng tiếng Trung.")

        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy tệp từ điển dịch tại {trans_dict_path}")
            # Cân nhắc trả về ma trận rỗng hoặc raise lỗi tùy theo logic chương trình
            # return torch.zeros(V1, V2)
            raise # Hoặc raise FileNotFoundError(f"Không tìm thấy tệp: {trans_dict_path}")
        except Exception as e:
            print(f"Lỗi không xác định xảy ra khi xử lý tệp {trans_dict_path}: {e}")
            # Cân nhắc trả về ma trận rỗng hoặc raise lỗi
            # return torch.zeros(V1, V2)
            raise # Hoặc raise e

        print(f"Đã tải từ điển dịch, tìm thấy {valid_pairs_count} cặp hợp lệ.")
        return trans_matrix

    def MutualInfo(self, anchor_feature, contrast_feature, mask, neg_mask, temperature):
        """Tính toán mất mát Mutual Information."""
        anchor_dot_contrast = torch.div(
            torch.matmul(F.normalize(anchor_feature, dim=1), F.normalize(contrast_feature, dim=1).T),
            temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits) * neg_mask
        sum_exp_logits = exp_logits.sum(1, keepdim=True)

        epsilon = 1e-10
        log_prob = logits - torch.log(sum_exp_logits + torch.exp(logits) + epsilon)
        mean_log_prob = -(mask * log_prob).sum()

        if torch.isnan(mean_log_prob) or torch.isinf(mean_log_prob):
            print("Cảnh báo: mean_log_prob là NaN hoặc Inf trong hàm MutualInfo.")
            mean_log_prob = torch.tensor(0.0, device=anchor_feature.device)

        return mean_log_prob

    def forward(self, x_bow_lang1, x_bow_lang2, cluster_info=None):
        """Lan truyền tiến."""
        theta_lang1, mu_lang1, logvar_lang1 = self.encoder_lang1(x_bow_lang1)
        theta_lang2, mu_lang2, logvar_lang2 = self.encoder_lang2(x_bow_lang2)
        beta_lang1 = self.compute_beta(self.word_embeddings_lang1, self.topic_embeddings_lang1)
        beta_lang2 = self.compute_beta(self.word_embeddings_lang2, self.topic_embeddings_lang2)
        
        tm_loss_lang1 = self.tm_loss(x_bow_lang1, theta_lang1, beta_lang1, mu_lang1, logvar_lang1, 'lang1')
        tm_loss_lang2 = self.tm_loss(x_bow_lang2, theta_lang2, beta_lang2, mu_lang2, logvar_lang2, 'lang2')
        
        contrast_loss = self.contrastive_loss(theta_lang1, theta_lang2, cluster_info, x_bow_lang1.shape[0]) if cluster_info else 0.0
        
        fea_lang1 = beta_lang1.T
        fea_lang2 = beta_lang2.T
        
        loss_MI = self.MutualInfo(fea_lang1, fea_lang2, self.pos_trans_mask_lang1, self.neg_trans_mask_lang1, temperature=self.temperature)
        loss_MI += self.MutualInfo(fea_lang2, fea_lang1, self.pos_trans_mask_lang2, self.neg_trans_mask_lang2, temperature=self.temperature)
        mask_sum = self.pos_trans_mask_lang1.sum() + self.pos_trans_mask_lang2.sum()
        loss_MI = loss_MI / (mask_sum + 1e-8)
        loss_MI = self.weight_MI * loss_MI

        total_loss = tm_loss_lang1 + tm_loss_lang2 + contrast_loss + loss_MI
        
        return {
            'total_loss': total_loss,
            'tm_loss_lang1': tm_loss_lang1,
            'tm_loss_lang2': tm_loss_lang2,
            'contrast_loss': contrast_loss,
            'loss_MI': loss_MI,
            'theta_lang1': theta_lang1,
            'theta_lang2': theta_lang2,
            'beta_lang1': beta_lang1,
            'beta_lang2': beta_lang2
        }

    def get_beta(self):
        """Lấy phân phối từ trên chủ đề (beta)."""
        beta_lang1 = self.compute_beta(self.word_embeddings_lang1, self.topic_embeddings_lang1)
        beta_lang2 = self.compute_beta(self.word_embeddings_lang2, self.topic_embeddings_lang2)
        return beta_lang1, beta_lang2

    def get_theta(self, bow, lang):
        """Lấy phân phối tài liệu trên chủ đề (theta)."""
        self.eval()
        with torch.no_grad():
            if lang == 'lang1':
                theta, _, _ = self.encoder_lang1(bow)
            elif lang == 'lang2':
                theta, _, _ = self.encoder_lang2(bow)
            else:
                raise ValueError(f"Ngôn ngữ không được hỗ trợ: {lang}")
        return theta