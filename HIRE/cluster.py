import torch
from torch import nn
import math
import torch.nn.functional as F
import numpy as np
import random
import copy

def gen_sim(sim1):
    random.seed(10)
    nnk = int(sim1.size()[0] * 0.06)
    nno = int(sim1.size()[0] * 0.06 * 1.5)
    dim = sim1.size()[0]
    final_sim = sim1 * 1.0
    sim = sim1 - torch.eye(dim).type(torch.FloatTensor)
    top = torch.rand((1, dim)).type(torch.FloatTensor)
    for i in range(dim):
        top[0, :] = sim[i, :]
        top20 = top.sort()[1][0]
        zero = torch.zeros(dim).type(torch.FloatTensor)
        zero[top20[-nnk:]] = 1.0
        sim[i, :] = top[0, :] * zero

    A = (sim > 0.0001).type(torch.FloatTensor)
    A = A * (A.t())
    A = A * sim
    sum_row = A.sum(1)
    aa = dim - (sum_row > 0).sum()
    kk = sum_row.sort()[1]
    res_ind = list(range(dim))
    for ind in range(aa):
        res_ind.remove(kk[ind])
    res_ind = random.sample(res_ind, dim - aa)
    ind_to_new_id = {}
    for i in range(dim - aa):
        ind_to_new_id[i] = res_ind[i]
    res_ind = (torch.from_numpy(np.asarray(res_ind))).type(torch.LongTensor)
    sim = sim[res_ind, :]
    sim = sim[:, res_ind]
    sim20 = {}
    dim = dim - aa
    top = torch.rand((1, dim)).type(torch.FloatTensor)
    for i in range(dim):
        top[0, :] = sim[i, :]
        top20 = top.sort()[1][0]
        zero = torch.zeros(dim).type(torch.FloatTensor)
        zero[top20[-nnk:]] = 1.0
        k = list(top20[-nnk:])
        sim20[i] = k
        sim[i, :] = top[0, :] * zero
    A = (sim > 0.0001).type(torch.FloatTensor)

    A = A * (A.t())
    A = A * sim
    sum_row = A.sum(1)

    sum_row = sum_row.pow(-0.5)
    sim = torch.diag(sum_row)
    A = A.mm(sim)
    A = sim.mm(A)
    alpha = 0.99
    manifold_sim = (1 - alpha) * torch.inverse(torch.eye(dim).type(torch.FloatTensor) - alpha * A)

    manifold20 = {}
    for i in range(dim):
        top[0, :] = manifold_sim[i, :]
        top20 = top.sort()[1][0]
        k = list(top20[-nno:])
        manifold20[i] = k
    for i in range(len(sim20)):
        aa = len(manifold20[i])
        zz = copy.deepcopy(manifold20[i])
        ddd = []
        for k in range(aa):
            if zz[k] in sim20[i]:
                sim20[i].remove(zz[k])
                manifold20[i].remove(zz[k])
                # print('k:', k)
                # print('zz[k]:', zz[k])
                # print('ind_to_new_id:', ind_to_new_id)
                ddd.append(ind_to_new_id[int(zz[k])])
        j = ind_to_new_id[int(i)]
        for l in ddd:
            final_sim[j, l] = 1.0
        for l in sim20[i]:
            final_sim[j, ind_to_new_id[int(l)]] = 0.0


    # final_sim = ((final_sim + final_sim.t()) > 0.1).type(torch.FloatTensor) - ((final_sim + final_sim.t()) < -0.1).type(torch.FloatTensor)
    f1 = (final_sim > 0.999).type(torch.FloatTensor)
    f1 = ((f1 + f1.t()) > 0.999).type(torch.FloatTensor)
    f2 = (final_sim < 0.0001).type(torch.FloatTensor)
    f2 = ((f2 + f2.t()) > 0.999).type(torch.FloatTensor)
    final_sim = final_sim * (1. - f2)
    final_sim = final_sim * (1. - f1) + f1
    
    return final_sim.sum(1)

def PairEnum(x,mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
    if mask is not None:
        xmask = mask.view(-1,1).repeat(1,x.size(1))
        #dim 0: #sample, dim 1:#feature 
        x1 = x1[xmask].view(-1,x.size(1))
        x2 = x2[xmask].view(-1,x.size(1))
    return x1,x2

class BCE(nn.Module):
    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()


class ClusterLoss():
    def __init__(self, device, num_classes, bce_type, cosine_threshold, topk):
        # super(NCLMemory, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.bce_type = bce_type
        self.costhre = cosine_threshold
        self.topk = topk
        self.bce = BCE()

    def compute_losses(self, inputs):
        bce_loss = 0.0
        device = self.device
        feat, output2 = inputs["x1"], inputs["preds1_u"]
        output2_bar = inputs["preds2_u"]
        label = inputs["labels"]

        num_s = (label < self.num_classes).sum()
        labels_s = label[:num_s]
        mask_lb = label < self.num_classes  # masked away label samples. only use unlabel samples for clustering
        
        prob2, prob2_bar = F.softmax(output2, dim=1), F.softmax(output2_bar, dim=1)

        rank_feat = (feat[~mask_lb]).detach()
        
        if self.bce_type == 'manifold':
            # default: cosine similarity with threshold
            feat_row, feat_col = PairEnum(F.normalize(rank_feat, dim=1))
            tmp_distance_ori = torch.matmul(feat_row, feat_col.T).cpu()
            #tmp_distance_ori = tmp_distance_ori.squeeze().cpu()

            target_ulb = gen_sim(tmp_distance_ori).cuda()
            target_ulb[target_ulb > self.costhre] = 1
            
        if self.bce_type == 'cos':
            # default: cosine similarity with threshold
            feat_row, feat_col = PairEnum(F.normalize(rank_feat, dim=1))
            tmp_distance_ori = torch.bmm(
                feat_row.view(feat_row.size(0), 1, -1),
                feat_col.view(feat_row.size(0), -1, 1)
            )
            tmp_distance_ori = tmp_distance_ori.squeeze()
            target_ulb = torch.zeros_like(tmp_distance_ori).float() - 1
            target_ulb[tmp_distance_ori > self.costhre] = 1
            
        elif self.bce_type == 'RK':
            # top-k rank statics
            rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
            rank_idx1, rank_idx2 = PairEnum(rank_idx)
            rank_idx1, rank_idx2 = rank_idx1[:, :self.topk], rank_idx2[:, :self.topk]
            rank_idx1, _ = torch.sort(rank_idx1, dim=1)
            rank_idx2, _ = torch.sort(rank_idx2, dim=1)
            rank_diff = rank_idx1 - rank_idx2
            rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
            target_ulb = torch.ones_like(rank_diff).float().to(device)
            target_ulb[rank_diff > 0] = -1

        prob1_ulb, _ = PairEnum(prob2[~mask_lb])
        _, prob2_ulb = PairEnum(prob2_bar[~mask_lb])

        bce_loss = self.bce(prob1_ulb, prob2_ulb, target_ulb)
        return bce_loss, target_ulb

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def reduce_dimension(features, mode, dim):
    if mode == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=dim)
        transformed_features = pca.fit_transform(features)
        fit_score = pca.explained_variance_ratio_.sum()
    elif mode == 'umap':
        import umap
        fit = umap.UMAP(n_components=dim)
        transformed_features = fit.fit_transform(features)
        fit_score = 0.0
    return transformed_features, fit_score