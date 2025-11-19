import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Models.fundus_swin_network import build_model as fundus_build_model
from Models.unetr import UNETR_base_3DNet


class ModalDisentangleLayer(nn.Module):
    def __init__(self, modality_count=2, sample_count=50, seed=1):
        super(ModalDisentangleLayer, self).__init__()

        self.sample_count = sample_count
        self.seed = seed

        phi = torch.ones(modality_count, requires_grad=True)
        self.phi = torch.nn.Parameter(phi)

    def forward(self, mu_list, var_list, eps=1e-8):
        t_sum = 0
        mu_t_sum = 0

        alpha = F.softmax(self.phi, dim=0)

        for idx, (mu, var) in enumerate(zip(mu_list, var_list)):
            T = 1 / (var + eps)

            t_sum += alpha[idx] * T
            mu_t_sum += mu * alpha[idx] * T

        mu = mu_t_sum / t_sum
        var = 1 / t_sum

        dim = mu.shape[1]
        batch_size = mu.shape[0]

        eps = self.generate_gaussian_noise(samples=(batch_size, self.sample_count), k=dim, seed=self.seed)
        eps = eps.unsqueeze(dim=-1).repeat(1, 1, 1, 128)

        disentangled_features = torch.unsqueeze(mu, dim=1) + torch.unsqueeze(var, dim=1)

        return disentangled_features

    def generate_gaussian_noise(self, samples, k, seed):
        if self.training:
            return torch.normal(torch.zeros(*samples, k), torch.ones(*samples, k)).cuda()
        else:
            return torch.normal(torch.zeros(*samples, k), torch.ones(*samples, k), generator=torch.manual_seed(seed)).cuda()


class JointProxyLayer(nn.Module):
    def __init__(self, x_dim, z_dim=256, beta=1e-2, sample_count=50, topk=1, num_classes=3, seed=1):
        super(JointProxyLayer, self).__init__()

        self.beta = beta
        self.sample_count = sample_count
        self.topk = topk
        self.num_classes = num_classes
        self.seed = seed
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Linear(x_dim, z_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(z_dim * 2, z_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(z_dim * 2, z_dim),
        )

        self.decoder_logits = nn.Linear(z_dim, num_classes)
        self.mlp_2d = nn.Sequential(nn.ReLU(), nn.Linear(144, num_classes), nn.Dropout(0.2), nn.ReLU())
        self.mlp_3d = nn.Sequential(nn.ReLU(), nn.Linear(216, num_classes), nn.Dropout(0.2), nn.ReLU())

        self.proxies = nn.Parameter(torch.empty([num_classes, z_dim * 2]))
        torch.nn.init.xavier_uniform_(self.proxies, gain=1.0)
        self.proxy_map = {"0": 0, "1": 1}

        self.alpha = nn.Parameter(torch.tensor(0.65))

    def generate_gaussian_noise(self, samples, K, seed):
        if self.training:
            return torch.normal(torch.zeros(*samples, K), torch.ones(*samples, K)).cuda()
        else:
            return torch.normal(torch.zeros(*samples, K), torch.ones(*samples, K), generator=torch.manual_seed(seed)).cuda()

    def encode_features(self, x):
        return self.encoder(x)

    def extract_proxy_parameters(self):
        mu_proxy = self.proxies[:, :self.z_dim]
        sigma_proxy = torch.nn.functional.softplus(self.proxies[:, self.z_dim:])
        return mu_proxy, sigma_proxy

    def calculate_entropy(self, logits):
        p = torch.softmax(logits, dim=1)
        log_p = torch.log_softmax(logits, dim=1)
        entropy = -torch.sum(p * log_p, dim=1)
        return entropy.mean()

    def forward(self, x, y=None):
        z = self.encode_features(x)

        mu_proxy, sigma_proxy = self.extract_proxy_parameters()

        eps_proxy = self.generate_gaussian_noise(samples=([self.num_classes, self.sample_count]), K=self.z_dim, seed=self.seed)

        z_proxy_sample = mu_proxy.unsqueeze(dim=1) + sigma_proxy.unsqueeze(dim=1) * eps_proxy
        z_proxy = z_proxy_sample

        z_norm = F.normalize(z, dim=1)
        z_proxy_norm = F.normalize(z_proxy)

        if not self.training:
            threshold = 0.5
            z_proxy_norm_expanded = z_proxy_norm.unsqueeze(0).expand(1, -1, -1, -1)

            att = torch.matmul(z_norm.unsqueeze(1), torch.transpose(z_proxy_norm_expanded, 2, 3))
            att = att.permute(0, 2, 1, 3).mean(dim=1)

            att_mean = torch.mean(att, dim=2)
            z_mean = torch.mean(z_norm, dim=2)

            pseudo_labels_att = torch.softmax(att_mean, dim=1)
            pseudo_labels_feat = torch.softmax(z_mean, dim=1)
            if pseudo_labels_feat.shape[1] == 144:
                pseudo_labels_feat = self.mlp_2d(pseudo_labels_feat)
            else:
                pseudo_labels_feat = self.mlp_3d(pseudo_labels_feat)

            pseudo_labels_combined = self.alpha * pseudo_labels_att + (1 - self.alpha) * pseudo_labels_feat

            confidence, labels = torch.max(pseudo_labels_combined, dim=1)
            mask = confidence > threshold

            if mask.sum().item() == 0:
                mask[confidence.argmax()] = True

            filtered_labels = labels[mask]

            proxy_indices = [self.proxy_map[str(int(label_item))] for label_item in filtered_labels]
            proxy_indices = torch.tensor(proxy_indices).long().cuda()

            mask = torch.zeros(att.size(0), att.size(1), dtype=torch.bool).cuda()
            mask[torch.arange(att.size(0)), proxy_indices] = True

            att_positive = torch.masked_select(att, mask.unsqueeze(-1)).view(att.size(0), -1)
            att_negative = torch.masked_select(att, ~mask.unsqueeze(-1)).view(att.size(0), -1)

            self_topk = 150
            att_topk_positive, _ = torch.topk(att_positive, self_topk, dim=1)
            att_topk_negative, _ = torch.topk(att_negative, self_topk, dim=1)

            att_positive_mean = torch.mean(att_topk_positive, dim=1)
            att_negative_mean = torch.mean(att_topk_negative, dim=1)

            proxy_loss = torch.mean(torch.exp(-att_positive_mean + att_negative_mean))

            entropy_loss = self.calculate_entropy(pseudo_labels_combined)

            mu_topk = mu_proxy.repeat(x.shape[0], 1, 1)
            sigma_topk = sigma_proxy.repeat(x.shape[0], 1, 1)
            z_topk = z

            return mu_topk, sigma_topk, proxy_loss, z_topk, entropy_loss

        else:
            z_proxy_norm_expanded = z_proxy_norm.unsqueeze(0).expand(16, -1, -1, -1)

            att = torch.matmul(z_norm.unsqueeze(1), torch.transpose(z_proxy_norm_expanded, 2, 3))
            att = att.permute(0, 2, 1, 3).mean(dim=1)

            proxy_indices = [self.proxy_map[str(int(y_item))] for y_item in y]
            proxy_indices = torch.tensor(proxy_indices).long().cuda()

            mask = torch.zeros(att.size(0), att.size(1), dtype=torch.bool).cuda()
            mask[torch.arange(att.size(0)), proxy_indices] = True

            att_positive = torch.masked_select(att, mask.unsqueeze(-1)).view(att.size(0), -1)
            att_negative = torch.masked_select(att, ~mask.unsqueeze(-1)).view(att.size(0), -1)

            self_topk = 150
            att_topk_positive, _ = torch.topk(att_positive, self_topk, dim=1)
            att_topk_negative, _ = torch.topk(att_negative, self_topk, dim=1)

            att_positive_mean = torch.mean(att_topk_positive, dim=1)
            att_negative_mean = torch.mean(att_topk_negative, dim=1)

            proxy_loss = torch.mean(torch.exp(-att_positive_mean + att_negative_mean))

        mu_topk = mu_proxy.repeat(x.shape[0], 1, 1)
        sigma_topk = sigma_proxy.repeat(x.shape[0], 1, 1)

        return mu_topk, sigma_topk, proxy_loss, z




class MutualInfoAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(MutualInfoAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ModalFusionAttention(nn.Module):
    def __init__(self, dim, dim_oct, dim_general, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., dropout=0.1):
        super(ModalFusionAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv_fundus = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_oct = nn.Linear(dim_oct, dim_oct * 3, bias=qkv_bias)
        self.qkv_general = nn.Linear(dim_general, dim_general * 3, bias=qkv_bias)
        self.norm = nn.LayerNorm(128)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_fundus = nn.Linear(dim, dim)
        self.proj_oct = nn.Linear(dim_oct, dim)
        self.proj_general = nn.Linear(8, 128)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(nn.ReLU(), nn.Linear(256, 32), nn.Dropout(0.2))

    def forward(self, x_2d, x_3d, x_global):
        B_2d, N_2d, C_2d = x_2d.shape
        B_3d, N_3d, C_3d = x_3d.shape
        B_gen, N_gen, C_gen = x_global.shape

        qkv_2d = self.qkv_fundus(x_2d).reshape(B_2d, N_2d, 3, self.num_heads, C_2d // self.num_heads).permute(2, 0, 3, 1, 4)
        q_2d, k_2d, v_2d = qkv_2d[0], qkv_2d[1], qkv_2d[2]

        qkv_3d = self.qkv_oct(x_3d).reshape(B_3d, N_3d, 3, self.num_heads, C_3d // self.num_heads).permute(2, 0, 3, 1, 4)
        q_3d, k_3d, v_3d = qkv_3d[0], qkv_3d[1], qkv_3d[2]

        qkv_general = self.qkv_general(x_global).reshape(B_gen, N_gen, 3, self.num_heads, C_gen // self.num_heads).permute(2, 0, 3, 1, 4)
        q_general, k_general, v_general = qkv_general[0], qkv_general[1], qkv_general[2]

        k_general = torch.mean(k_general, dim=2)
        k_3d = torch.mean(k_3d, dim=2)
        k_2d = torch.mean(k_2d, dim=2)
        q_general = torch.mean(q_general, dim=2)

        k_combined = self.fc(torch.cat((k_general, k_3d, k_2d), dim=2))

        attn_global = (q_general @ k_combined.transpose(-1, -2)) * self.scale
        attn_global = attn_global.softmax(dim=-1)
        attn_global = self.attn_drop(attn_global)

        v_general = torch.mean(v_general, dim=2)
        v_3d = torch.mean(v_3d, dim=2)
        v_2d = torch.mean(v_2d, dim=2)

        v_combined = self.fc(torch.cat((v_general, v_3d, v_2d), dim=2))

        attn_global_x = (attn_global @ v_combined).transpose(1, 2)

        attn_global_x = self.proj_general(attn_global_x)
        x_global = self.norm(self.dropout(attn_global_x))

        return x_global

class MultimodalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultimodalSelfAttention, self).__init__()
        self.attention = MutualInfoAttention(embed_dim, num_heads)

    def forward(self, x):
        return self.attention(x)

class MultimodalFusionAttention(nn.Module):
    def __init__(self, embed_dim, dim_oct, dim_general, num_heads):
        super(MultimodalFusionAttention, self).__init__()
        self.attention = ModalFusionAttention(embed_dim, dim_oct, dim_general, num_heads)

    def forward(self, x, y, z):
        return self.attention(x, y, z)

def KL_divergence_between_distributions(q_distr, p_distr):
    mu_q, sigma_q = q_distr
    mu_p, sigma_p = p_distr
    k = mu_q.size(1)

    mu_diff = mu_p - mu_q
    mu_diff_sq = torch.mul(mu_diff, mu_diff)
    logdet_sigma_q = torch.sum(2 * torch.log(torch.clamp(sigma_q, min=1e-8)), dim=1)
    logdet_sigma_p = torch.sum(2 * torch.log(torch.clamp(sigma_p, min=1e-8)), dim=1)

    fs = torch.sum(torch.div(sigma_q ** 2, sigma_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, sigma_p ** 2), dim=1)
    two_kl = fs - k + logdet_sigma_p - logdet_sigma_q
    return two_kl * 0.5



class IntegratedDisentangleLayer(nn.Module):
    def __init__(self, embed_dim, embed_dim_3d, dim_general, num_heads, dropout=0.1, dim_pid=256):
        super(IntegratedDisentangleLayer, self).__init__()

        self.self_attn_3d = MultimodalSelfAttention(embed_dim_3d, num_heads)
        self.self_attn_2d = MultimodalSelfAttention(embed_dim, num_heads)
        self.fc = nn.Sequential(nn.ReLU(), nn.Linear(768, 1024), nn.ReLU())
        self.dropout = nn.Dropout(dropout)
        self.norm_2d = nn.LayerNorm(embed_dim)
        self.norm_3d = nn.LayerNorm(embed_dim_3d)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fusion = MultimodalFusionAttention(embed_dim, embed_dim_3d, dim_general, num_heads)

    def forward(self, x_2d, x_3d, distribution):
        x_2d_attn = self.self_attn_2d(x_2d)
        x_3d_attn = self.self_attn_3d(x_3d)
        x_3d_attn = self.fc(x_3d_attn)

        x_fusion = self.fusion(x_2d, x_3d, distribution)
        x_3d_combined = self.avgpool(x_3d_attn.transpose(1, 2))
        x_2d_combined = self.avgpool(x_2d_attn.transpose(1, 2))
        x_fusion = self.avgpool(x_fusion.transpose(1, 2))

        return x_2d_combined, x_3d_combined, x_fusion


class MutualInfoEstimator(nn.Module):
    def __init__(self, dim=128):
        super(MutualInfoEstimator, self).__init__()
        self.dim = dim
        self.mimin_glob = ConditionalLowerBound(self.dim * 2, self.dim)
        self.mimin = ConditionalLowerBound(self.dim, self.dim)

    def forward(self, histology, pathways, global_embed):
        mimin = self.mimin(histology, pathways)
        mimin += self.mimin_glob(torch.cat((histology, pathways), dim=1), global_embed)
        return mimin

    def calculate_learning_loss(self, histology, pathways, global_embed):
        mimin_loss = self.mimin.calculate_learning_loss(histology, pathways)
        mimin_loss += self.mimin_glob.calculate_learning_loss(torch.cat((histology, pathways), dim=1), global_embed).mean()
        return mimin_loss


class ConditionalLowerBound(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size=512):
        super(ConditionalLowerBound, self).__init__()

        if hidden_size is None:
            self.p_mu = nn.Linear(x_dim, y_dim)
        else:
            self.p_mu = nn.Sequential(nn.Linear(x_dim, int(hidden_size)), nn.ReLU(), nn.Linear(int(hidden_size), y_dim))

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        return mu, 0

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        positive = - (mu - y_samples) ** 2 / 2.

        prediction_1 = mu.unsqueeze(1)
        y_samples_1 = y_samples.unsqueeze(0)

        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2.

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def calculate_log_likelihood(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2).sum(dim=1).mean(dim=0)

    def calculate_learning_loss(self, x_samples, y_samples):
        return - self.calculate_log_likelihood(x_samples, y_samples)


class IMDR(nn.Module):
    def __init__(self, num_classes, modalities, classifier_dims, args):

        super(IMDR, self).__init__()
        self.modalities = modalities
        self.num_classes = num_classes
        self.mode = args.mode
        dropout_rate = 0.35
        self.fundus_embedding_dim = 1024
        self.oct_embedding_dim = 768
        self.shared_dim = 256

        self.num_latent_classes = 2
        self.topk_fundus = 1
        self.topk_oct = 1
        self.num_samples = 750
        self.seed = 23
        self.num_heads = 8

        self.fundus_transformer = fundus_build_model()

        self.oct_transformer = UNETR_base_3DNet(num_classes=self.num_classes)

        self.fc_fundus = nn.Sequential(nn.ReLU(), nn.Linear(128, 1024), nn.ReLU())

        self.classifier = nn.Sequential(nn.ReLU(), nn.Linear(2304, 64), nn.ReLU(),
                                        nn.Linear(64, self.num_classes))

        fundus_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=self.num_heads,
                                                          dim_feedforward=512, dropout=dropout_rate,
                                                          activation='relu')

        self.fundus_transformer_layer = nn.TransformerEncoder(fundus_encoder_layer, num_layers=2)

        oct_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=self.num_heads,
                                                       dim_feedforward=512, dropout=dropout_rate,
                                                       activation='relu')
        self.oct_transformer_layer = nn.TransformerEncoder(oct_encoder_layer, num_layers=2)

        self.ProxyInstanceBank_fundus = JointProxyLayer(self.fundus_embedding_dim, num_classes=self.num_latent_classes,
                                                        topk=self.topk_fundus, sample_num=self.num_samples, seed=self.seed)

        self.ProxyInstanceBank_oct = JointProxyLayer(self.oct_embedding_dim, num_classes=self.num_latent_classes,
                                                     topk=self.topk_oct, sample_num=self.num_samples, seed=self.seed)

        self.PosteriorEmbedding = ModalDisentangleLayer(modality_num=2, sample_num=self.num_samples, seed=self.seed)

        self.DisentangleEmbedding = IntegratedDisentangleLayer(self.fundus_embedding_dim, self.oct_embedding_dim,
                                                               self.shared_dim, self.num_heads, dropout=dropout_rate,
                                                               dim_pid=256)

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.global_avgpool = nn.AdaptiveAvgPool1d(1)

        self.MutualInfoEstimator = MutualInfoEstimator(self.fundus_embedding_dim)

        self.args = args

    def compute_kl_loss(self, mu, std):

        prior_distr = torch.zeros_like(mu), torch.ones_like(std)
        posterior_distr = mu, std

        kl_divergence = torch.mean(KL_divergence_between_distributions(posterior_distr, prior_distr))

        return torch.mean(kl_divergence)

    def compute_test_loss(self, loss1, proxy_ib_loss, proxy_loss_fundus, proxy_loss_oct, mi_loss, entropy_loss):

        loss = loss1 + proxy_ib_loss + (proxy_loss_fundus + proxy_loss_oct) * 0.8 + 0.001 * mi_loss + entropy_loss * 0.01
        return loss

    def compute_train_loss(self, loss1, proxy_ib_loss, proxy_loss_fundus, proxy_loss_oct, mi_loss):

        loss = loss1 + proxy_ib_loss + (proxy_loss_fundus + proxy_loss_oct) * 0.5 + 0.001 * mi_loss
        return loss

    def forward(self, inputs, labels, epoch):
        x_fundus, fundus_features = self.fundus_transformer(inputs[0])
        x_oct, oct_features = self.oct_transformer(inputs[1])


        if not self.training:
            mu_fundus, sigma_fundus, proxy_loss_fundus, z_fundus, entropy_loss = self.ProxyInstanceBank_fundus(x_fundus, y=labels)
            mu_oct, sigma_oct, proxy_loss_oct, z_oct, entropy_loss = self.ProxyInstanceBank_oct(x_oct, y=labels)
        else:
            mu_fundus, sigma_fundus, proxy_loss_fundus, z_fundus = self.ProxyInstanceBank_fundus(x_fundus, y=labels)
            mu_oct, sigma_oct, proxy_loss_oct, z_oct = self.ProxyInstanceBank_oct(x_oct, y=labels)

        mu_list = [mu_fundus, mu_oct]
        var_list = [sigma_fundus, sigma_oct]

        poe_features = self.PosteriorEmbedding(mu_list, var_list)
        poe_embed = torch.mean(poe_features, dim=1).view(16, -1)

        B, N, D = poe_embed.shape
        global_fusion = self.fc_fundus(poe_embed.reshape(B, -1))

        x_fundus, x_oct, x_fusion = self.DisentangleEmbedding(x_fundus, x_oct, poe_embed)
        x_fusion = self.fc_fundus(x_fusion.squeeze(2))

        mi_loss = self.MutualInfoEstimator.learning_loss(x_fundus.squeeze(2), x_oct.squeeze(2), x_fusion)

        combined_features = torch.cat([x_fundus.squeeze(2), global_fusion, x_oct.squeeze(2)], 1)

        pred = self.classifier(combined_features)

        loss1 = self.cross_entropy_loss(pred, labels)

        proxy_ib_loss = 0.01 * self.compute_kl_loss(mu_fundus, sigma_fundus) + 0.01 * self.compute_kl_loss(mu_oct, sigma_oct)

        if not self.training:
            loss = self.compute_test_loss(loss1, proxy_ib_loss, proxy_loss_fundus, proxy_loss_oct, mi_loss, entropy_loss)
        else:
            loss = self.compute_train_loss(loss1, proxy_ib_loss, proxy_loss_fundus, proxy_loss_oct, mi_loss)

        loss = torch.mean(loss)
        return pred, loss, combined_features


class TwoDTransformer(nn.Module):
    def __init__(self, num_classes, modalities, classifier_dims, args):


        super(TwoDTransformer, self).__init__()
        self.modalities = modalities
        self.num_classes = num_classes
        self.mode = args.mode

        self.fundus_transformer = fundus_build_model()

        self.classifier = nn.Sequential(nn.ReLU(), nn.Linear(1024, 64), nn.ReLU(),
                                        nn.Linear(64, self.num_classes))

        self.fc_fundus = nn.Sequential(nn.ReLU(), nn.Linear(1024, 768), nn.ReLU())

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.args = args

    def forward(self, inputs, labels):
        x_fundus, fundus_features = self.fundus_transformer(inputs[0])
        fundus_features = self.fc_fundus(fundus_features)

        pred = self.classifier(fundus_features)

        return pred, fundus_features


class ThreeDTransformer(nn.Module):
    def __init__(self, num_classes, modalities, classifier_dims, args):

        super(ThreeDTransformer, self).__init__()
        self.modalities = modalities
        self.num_classes = num_classes
        self.mode = args.mode

        self.oct_transformer = UNETR_base_3DNet(num_classes=self.num_classes)

        self.classifier = nn.Sequential(nn.ReLU(), nn.Linear(768, 64), nn.ReLU(),
                                        nn.Linear(64, self.num_classes))

        self.args = args

    def forward(self, inputs, labels):
        x_oct, oct_features = self.oct_transformer(inputs[1])

        pred = self.classifier(oct_features)

        return pred, oct_features

