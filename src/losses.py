import torch
import torch.nn as nn


class CCLoss(nn.Module):
    def __init__(self, gamma, temperature=0.05, K=100, num_classes=21):
        super(CCLoss, self).__init__()
        self.gamma = gamma
        self.temperature = temperature
        self.K = K
        self.num_classes = num_classes

    def forward(self, features, labels=None, sup_logits=None):
        device = features.device
        
        bs = features.shape[0] - self.K        
        feature_sim = torch.matmul(features[:bs], features.T) / self.temperature
        logits_con = torch.cat(( sup_logits, feature_sim), dim=1)
        logits_max, _ = torch.max(logits_con, dim=1, keepdim=True)
        logits = logits_con - logits_max.detach()

        labels = labels.contiguous().view(-1, 1)
        con_mask = torch.eq(labels[:bs], labels.T).float().to(device)
        logits_mask = torch.scatter(torch.ones_like(con_mask),1,torch.arange(bs).view(-1, 1).to(device),0)
        e_mask = con_mask * logits_mask

        one_hot_label = torch.nn.functional.one_hot(labels[:bs,].view(-1,), num_classes=self.num_classes).to(torch.float32)
        df_mask = torch.cat((one_hot_label, e_mask * self.gamma), dim=1)

        logits_mask = torch.cat((torch.ones(bs, self.num_classes).to(device), logits_mask), dim=1)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (df_mask * log_prob).sum(1) / df_mask.sum(1)
        
        loss = - mean_log_prob_pos.mean()
        return loss



