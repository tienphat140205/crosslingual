import torch
import torch.nn as nn
import torch.nn.functional as F
from ot import sinkhorn
from models.networks.Encoder import MLPEncoder

class DualTopic(nn.Module):
    def __init__(self, vocab_size_en, vocab_size_cn, num_topics, hidden_dim, word_embeddings_en, word_embeddings_cn, dropout=0.3, tau=1.0, lambda_contrast=1.0, gamma_align=1.0, epsilon=0.1):
        super(DualTopic, self).__init__()
        self.num_topics = num_topics
        self.tau = tau
        self.lambda_contrast = lambda_contrast
        self.gamma_align = gamma_align
        self.epsilon = epsilon

        self.encoder_en = MLPEncoder(vocab_size_en, num_topics, hidden_dim, dropout)
        self.encoder_cn = MLPEncoder(vocab_size_cn, num_topics, hidden_dim, dropout)

        self.word_embeddings_en = nn.Parameter(torch.from_numpy(word_embeddings_en).float(), requires_grad=False)
        self.word_embeddings_cn = nn.Parameter(torch.from_numpy(word_embeddings_cn).float(), requires_grad=False)

        self.topic_embeddings = nn.Parameter(torch.randn(num_topics, self.word_embeddings_en.shape[1]))

    def compute_beta(self, word_embeddings):
        dist = torch.cdist(word_embeddings.unsqueeze(0), self.topic_embeddings.unsqueeze(0), p=2).squeeze(0)
        beta = torch.exp(-dist / self.tau)
        beta = beta / beta.sum(dim=0, keepdim=True)
        return beta.transpose(0, 1)

    def tm_loss(self, x_bow, theta, beta, mu, logvar):
        recon = torch.matmul(theta, beta)
        recon_loss = -torch.sum(x_bow * torch.log_softmax(recon, dim=-1), dim=-1).mean()
        kl_loss = 0.5 * torch.sum(logvar.exp() + mu.pow(2) - 1 - logvar, dim=-1).mean()
        return recon_loss + kl_loss

    def contrastive_loss(self, theta_en, theta_cn, cluster_info, batch_size):
        loss = 0.0
        count = 0
        for i in range(batch_size):
            anchor_theta = theta_en[i] if i % 2 == 0 else theta_cn[i]
            anchor_cluster = cluster_info[i][0][0] if cluster_info[i] else None
            if not anchor_cluster:
                continue
            pos_sim = 0.0
            neg_sum = 0.0
            pos_count = 0
            for j in range(batch_size):
                if i == j:
                    continue
                other_theta = theta_en[j] if j % 2 == 0 else theta_cn[j]
                other_cluster = cluster_info[j][0][0] if cluster_info[j] else None
                sim = F.cosine_similarity(anchor_theta.unsqueeze(0), other_theta.unsqueeze(0)).squeeze()
                sim_exp = torch.exp(sim / self.tau)
                if other_cluster == anchor_cluster:
                    pos_sim += sim_exp
                    pos_count += 1
                neg_sum += sim_exp
            if pos_count > 0:
                loss += -torch.log(pos_sim / (pos_sim + neg_sum + 1e-8))
                count += 1
        return loss / (count + 1e-8)

    def alignment_loss(self):
        t1 = self.topic_embeddings
        t2 = self.topic_embeddings
        cos_sim = F.cosine_similarity(t1.unsqueeze(1), t2.unsqueeze(0), dim=-1)
        C = 1 - cos_sim
        a = torch.ones(self.num_topics, device=C.device) / self.num_topics
        b = torch.ones(self.num_topics, device=C.device) / self.num_topics
        Q = sinkhorn(a, b, C, self.epsilon)
        ot_loss = torch.sum(Q * C)
        return ot_loss

    def forward(self, x_bow_en, x_bow_cn, cluster_info=None):
        theta_en, mu_en, logvar_en = self.encoder_en(x_bow_en)
        theta_cn, mu_cn, logvar_cn = self.encoder_cn(x_bow_cn)
        beta_en = self.compute_beta(self.word_embeddings_en)
        beta_cn = self.compute_beta(self.word_embeddings_cn)
        tm_loss_en = self.tm_loss(x_bow_en, theta_en, beta_en, mu_en, logvar_en)
        tm_loss_cn = self.tm_loss(x_bow_cn, theta_cn, beta_cn, mu_cn, logvar_cn)
        contrast_loss = self.contrastive_loss(theta_en, theta_cn, cluster_info, x_bow_en.shape[0]) if cluster_info else 0.0
        align_loss = self.alignment_loss()
        total_loss = tm_loss_en + tm_loss_cn + self.lambda_contrast * contrast_loss + self.gamma_align * align_loss
        return {
            'total_loss': total_loss,
            'tm_loss_en': tm_loss_en,
            'tm_loss_cn': tm_loss_cn,
            'contrast_loss': contrast_loss,
            'align_loss': align_loss,
            'theta_en': theta_en,
            'theta_cn': theta_cn,
            'beta_en': beta_en,
            'beta_cn': beta_cn
        }