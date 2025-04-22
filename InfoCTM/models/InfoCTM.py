import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.Encoder import Encoder


class InfoCTM(nn.Module):
    def __init__(self, args, trans_e2c):
        super().__init__()

        self.args = args

        # FIXED: Added missing vocabulary properties
        self.vocab_en = args.vocab_en
        self.vocab_cn = args.vocab_cn

        # Fixed incomplete assignments - Added proper initialization
        # Comment: Added proper initialization for cluster information parameters
        self.clusterinfo_en = args.cluster_en # Will be set during forward pass
        self.clusterinfo_cn = args.cluster_cn # Will be set during forward pass

        self.mask_en_to_cn = args.mask_en_to_cn
        self.mask_cn_to_en = args.mask_cn_to_en

        self.trans_e2c = torch.as_tensor(trans_e2c).float()
        self.trans_e2c = nn.Parameter(self.trans_e2c, requires_grad=False)
        self.trans_c2e = self.trans_e2c.T
        
        self.vocab_size_en = len(self.vocab_en)
        self.vocab_size_cn = len(self.vocab_cn)

        self.num_topic = args.num_topic
        self.temperature = args.temperature
        # Comment: Added missing tau parameter for compute_beta method
        self.tau = args.tau if hasattr(args, 'tau') else 1.0  
        # Comment: Added lambda parameter for contrastive loss
        self.lambda_contrast = args.lambda_contrast 
        self.weight_MI = args.weight_MI
        self.infonce_alpha = args.infoncealpha  # FIXED: Renamed for consistency
        
        self.encoder_en = Encoder(args.vocab_size_en, args.num_topic, args.en1_units, args.dropout)
        self.encoder_cn = Encoder(args.vocab_size_cn, args.num_topic, args.en1_units, args.dropout)

        self.a = 1 * np.ones((1, int(args.num_topic))).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T), requires_grad=False)
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / args.num_topic))).T + (1.0 / (args.num_topic * args.num_topic)) * np.sum(1.0 / self.a, 1)).T), requires_grad=False)
        
        self.decoder_bn_en = nn.BatchNorm1d(args.vocab_size_en, affine=True)
        self.decoder_bn_en.weight.requires_grad = False
        self.decoder_bn_cn = nn.BatchNorm1d(args.vocab_size_cn, affine=True)
        self.decoder_bn_cn.weight.requires_grad = False

        # Comment: Fixed embeddings initialization with proper parameters
        self.pretrain_word_embeddings_en = nn.Parameter(
            torch.tensor(args.pretrain_word_embeddings_en, dtype=torch.float32), 
            requires_grad=True
        )
        self.pretrain_word_embeddings_cn = nn.Parameter(
            torch.tensor(args.pretrain_word_embeddings_cn, dtype=torch.float32), 
            requires_grad=True
        )


        # Comment: Added proper initialization for topic embeddings
        topic_embed_dim = args.pretrain_word_embeddings_en.shape[1]
        self.topic_embedding_en = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty((args.num_topic, topic_embed_dim))), 
            requires_grad=True
        )
        self.topic_embedding_cn = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty((args.num_topic, topic_embed_dim))), 
            requires_grad=True
        )
  
        # FIXED: Corrected duplicate projection layers definition
        self.prj_beta_en = nn.Sequential(
            nn.Linear(args.vocab_size_en, 384),
            nn.Dropout(args.dropout), 
        )
        # FIXED: Renamed from prj_beta_en to prj_beta_cn
        self.prj_beta_cn = nn.Sequential(
            nn.Linear(args.vocab_size_cn, 384),
            nn.Dropout(args.dropout), 
        )
        
    # FIXED: Renamed from acompute_beta to compute_beta to match method calls
    def compute_beta(self, word_embeddings, topic_embeddings):
        dist = torch.cdist(word_embeddings.unsqueeze(0), topic_embeddings.unsqueeze(0), p=2).squeeze(0)
        # Clamp extreme values and add stability
        dist_scaled = torch.clamp(dist / max(self.tau, 1e-4), -50, 50)
        beta = torch.exp(-dist_scaled)
        # Add small epsilon to avoid division by zero
        beta = beta / (beta.sum(dim=0, keepdim=True) + 1e-10)
        return beta.transpose(0, 1)

    def MutualInfo(self, anchor_feature, contrast_feature, mask, neg_mask, temperature):
        # Higher epsilon for better stability
        eps = 1e-6
        
        # Check for and fix any NaN inputs
        if torch.isnan(anchor_feature).any() or torch.isnan(contrast_feature).any():
            print("Warning: NaN detected in features")
            anchor_feature = torch.where(torch.isnan(anchor_feature), torch.zeros_like(anchor_feature), anchor_feature)
            contrast_feature = torch.where(torch.isnan(contrast_feature), torch.zeros_like(contrast_feature), contrast_feature)
        
        # Safe normalization
        anchor_norm = F.normalize(anchor_feature, dim=1, eps=eps)
        contrast_norm = F.normalize(contrast_feature, dim=1, eps=eps)
        
        # Calculate similarity with temperature scaling
        anchor_dot_contrast = torch.matmul(anchor_norm, contrast_norm.T) / max(temperature, 1e-4)
        
        # Numerical stability: subtract max value
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Safe exponentiation
        exp_logits = torch.exp(torch.clamp(logits, -50, 50)) * neg_mask
        sum_exp_logits = exp_logits.sum(1, keepdim=True)
        
        # Safe logarithm
        denominator = sum_exp_logits + torch.exp(torch.clamp(logits, -50, 50)) + eps
        log_prob = logits - torch.log(denominator)
        
        # Calculate final loss
        mean_log_prob = -(mask * log_prob).sum()
        
        # Check for NaN in output
        if torch.isnan(mean_log_prob):
            print("Warning: NaN detected in MutualInfo output")
            return torch.tensor(0.0, device=anchor_feature.device)
            
        return mean_log_prob

    # Comment: Fixed contrastive loss function formatting and implementation    
    def contrastive_loss(self, theta_lang1, theta_lang2, cluster_info_lang1, cluster_info_lang2):
        """Calculate contrastive loss based on cluster information."""
        batch_size = theta_lang1.size(0)
        theta_all = torch.cat([theta_lang1, theta_lang2], dim=0)
        
        # Convert cluster info tensors to appropriate format
        if isinstance(cluster_info_lang1, torch.Tensor) and isinstance(cluster_info_lang2, torch.Tensor):
            cluster_all = torch.cat([cluster_info_lang1, cluster_info_lang2], dim=0)
        else:
            # Handle if cluster_info is a list of tensors or other formats
            cluster_all = []
            for c1, c2 in zip(cluster_info_lang1, cluster_info_lang2):
                if isinstance(c1, torch.Tensor) and isinstance(c2, torch.Tensor):
                    cluster_all.extend([c1, c2])
                else:
                    cluster_all.extend([torch.tensor(c1, device=theta_lang1.device), 
                                       torch.tensor(c2, device=theta_lang1.device)])
            cluster_all = torch.stack(cluster_all) if cluster_all else None
        
        if cluster_all is None:
            return torch.tensor(0.0, device=theta_lang1.device)
            
        theta_norm = F.normalize(theta_all, dim=-1)
        sim_matrix = torch.matmul(theta_norm, theta_norm.T) / self.temperature
        sim_exp = torch.exp(sim_matrix)
        
        eye_mask = torch.eye(2 * batch_size, device=theta_all.device, dtype=torch.bool)
        sim_exp = sim_exp * (~eye_mask).float()
        
        # Create positive mask based on cluster information
        pos_mask = (cluster_all.unsqueeze(0) == cluster_all.unsqueeze(1)).float()
        pos_mask = pos_mask * (~eye_mask).float()
        
        pos_sim = torch.sum(sim_exp * pos_mask, dim=1)
        neg_sum = torch.sum(sim_exp, dim=1) - sim_exp.diag()
        
        loss_per_anchor = -torch.log(pos_sim / (pos_sim + neg_sum + 1e-8) + 1e-8)
        valid_anchors = (pos_sim > 0).float()
        count = torch.sum(valid_anchors)
        total_loss = torch.sum(loss_per_anchor * valid_anchors)
        return (total_loss / (count + 1e-8)) * self.lambda_contrast 

    def get_beta(self):
        beta_en = self.compute_beta(self.pretrain_word_embeddings_en, self.topic_embedding_en)
        beta_cn = self.compute_beta(self.pretrain_word_embeddings_cn, self.topic_embedding_cn)
        return beta_en, beta_cn

    def get_theta(self, x, lang):
        theta, mu, logvar = getattr(self, f'encoder_{lang}')(x)
        if self.training:
            return theta, mu, logvar
        else:
            return mu

    def decode(self, theta, beta, lang):
        bn = getattr(self, f'decoder_bn_{lang}')
        d1 = F.softmax(bn(torch.matmul(theta, beta)), dim=1)
        return d1

    def loss_function(self, recon_x, x, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.num_topic)

        RECON = -(x * (recon_x + 1e-10).log()).sum(1)

        LOSS = (RECON + KLD).mean()
        return LOSS


    def csim(self, beta_en, beta_cn):
        # FIXED: Translated comments to English
        # Project beta_en and beta_cn into common space
        pbeta_en = self.prj_beta_en(beta_en)  # [K, 384]
        pbeta_cn = self.prj_beta_cn(beta_cn)  # [K, 384]

        # Calculate cosine similarity matrix
        csim_matrix = (pbeta_en @ pbeta_cn.T) / (pbeta_en.norm(keepdim=True, dim=-1) @ pbeta_cn.norm(keepdim=True, dim=-1).T + 1e-8)
        
        # Convert to exponential form
        csim_matrix = torch.exp(csim_matrix)
        
        # Normalize so each row sums to 1
        csim_matrix = csim_matrix / (csim_matrix.sum(dim=1, keepdim=True) + 1e-8)
        
        # Return -log of probability matrix
        return -csim_matrix.log()

    def InfoNce(self, beta_en, beta_cn):
        # FIXED: Translated comments to English
        # Calculate -log(p) matrix
        log_p_matrix = self.csim(beta_en, beta_cn)
        loss = log_p_matrix.diag().mean()
        return loss
        
    def forward(self, x_en, x_cn, cluster_info=None):
        theta_en, mu_en, logvar_en = self.get_theta(x_en, lang='en')
        theta_cn, mu_cn, logvar_cn = self.get_theta(x_cn, lang='cn')

        beta_en, beta_cn = self.get_beta()

        loss = 0.
        tmp_rst_dict = dict()

        x_recon_en = self.decode(theta_en, beta_en, lang='en')
        x_recon_cn = self.decode(theta_cn, beta_cn, lang='cn')
        loss_en = self.loss_function(x_recon_en, x_en, mu_en, logvar_en)
        loss_cn = self.loss_function(x_recon_cn, x_cn, mu_cn, logvar_cn)

        loss = loss_en + loss_cn
        tmp_rst_dict['loss_en'] = loss_en
        tmp_rst_dict['loss_cn'] = loss_cn

        fea_en = beta_en.T
        fea_cn = beta_cn.T
        
        # Create neg mask by inverting positive mask (1 - positive_mask)
        neg_mask_en_to_cn = 1.0 - self.mask_en_to_cn
        neg_mask_cn_to_en = 1.0 - self.mask_cn_to_en
        loss_TAMI = self.MutualInfo(fea_en, fea_cn, self.mask_en_to_cn, neg_mask_en_to_cn, temperature=self.temperature)
        loss_TAMI += self.MutualInfo(fea_cn, fea_en, self.mask_cn_to_en, neg_mask_cn_to_en, temperature=self.temperature)

        loss_TAMI = loss_TAMI / (self.mask_en_to_cn.sum() + self.mask_cn_to_en.sum())

        loss_TAMI = self.weight_MI * loss_TAMI
        loss += loss_TAMI

        # Comment: Fixed contrastive loss calculation
        loss_contrastive = 0.0
        if cluster_info and 'cluster_en' in cluster_info and 'cluster_cn' in cluster_info:
            cluster_info_en = cluster_info['cluster_en']
            cluster_info_cn = cluster_info['cluster_cn']
            # Only calculate contrastive loss if cluster info is available
            if cluster_info_en is not None and cluster_info_cn is not None:
                loss_contrastive = self.contrastive_loss(theta_en, theta_cn, cluster_info_en, cluster_info_cn)
                loss += loss_contrastive

        # FIXED: Calculate InfoNce loss and fixed syntax error
        infonce = self.InfoNce(beta_en, beta_cn)
        loss += infonce * self.infonce_alpha
        
        tmp_rst_dict['loss_TAMI'] = loss_TAMI
        tmp_rst_dict['loss_contrastive'] = loss_contrastive
        tmp_rst_dict['loss_infonce'] = infonce        
        rst_dict = {
            'loss': loss,
        }

        rst_dict.update(tmp_rst_dict)

        return rst_dict