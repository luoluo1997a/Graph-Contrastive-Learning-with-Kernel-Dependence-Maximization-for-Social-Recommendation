# -*- coding: utf-8 -*-

import os
import pickle
from recbole_gnn.DGI.dgi import DGI

import numpy as np
import dgl
# from recbole_gnn.DGI.dgi import DGI
import torch.nn as nn
import time
import torch
import torch as t
modelUTCStr = str(int(time.time()))[4:]

import torch.nn.functional as F

from recbole_gnn.model.general_recommender import LightGCN
from recbole_gnn.ToolScripts.utils import generate_sp_ont_hot
from recbole_gnn.ToolScripts.utils import sparse_mx_to_torch_sparse_tensor
from recbole_gnn.model.abstract_recommender import GeneralGraphRecommender
from recbole_gnn.ToolScripts.utils import loadData
from recbole_gnn.ToolScripts.TimeLogger import log

import scipy.sparse as sp
device_gpu = t.device("cuda")
# from recbole_gnn import MyData


class CLKDM(LightGCN):

    def getData(self, config,dataset):
        trustMat,trainmat = loadData(dataset, config['rate'])
        # assert a.nnz == (trainMat.nnz + testMat.nnz + validMat.nnz)
        DIR = '/home/nixuelian/anaconda3/envs/test/lib/python3.8/site-packages/recbole/dataset_example/ml-100k'
        adj_path_4 = DIR + '/{0}_community.pkl'.format(config['rate'])
        adj_path_2 = DIR + '/{0}_feihaoyou.pkl'.format(config['rate'])
        adj_path_3 = DIR + '/{0}_feihaoyou_shequnei.pkl'.format(config['rate'])
        adj_path_5 = DIR + '/{0}_feiling_shequnei.pkl'.format(config['rate'])
        adj_path_6 = DIR + '/{0}_feiling_shequwai.pkl'.format(config['rate'])

        log(adj_path_4)
        with open(adj_path_4, 'rb') as fs:
            community = pickle.load(fs)

        with open(adj_path_2, 'rb') as fs:
            feihaoyou_mat = pickle.load(fs)
        with open(adj_path_3, 'rb') as fs:
            feihaoyou_shequnei_mat = pickle.load(fs)
        with open(adj_path_5, 'rb') as fs:
            feiling_shequnei = pickle.load(fs)
        with open(adj_path_6, 'rb') as fs:
            feiling_shequwai = pickle.load(fs)
        return  trustMat, community,trainmat,feihaoyou_mat, feihaoyou_shequnei_mat,feiling_shequnei,feiling_shequwai

        # 初始化参数

    def __init__(self, config, dataset):
        super(CLKDM, self).__init__(config, dataset)
        trust, community,train,feihaoyou, feihaoyou_shequnei,feiling_shequnei,feiling_shequwai= self.getData(config,dataset)

        self.cl_rate = config['lambda']
        self.eps = config['eps']
        self.temperature = config['temperature']
        self.hSICLoss = HSICLoss(config,dataset)
        self.hsic =  config['hsic']
        self.hsic_xx_lamb = config['hsic_xx_lamb']


        self.dgi_path = self.preTrain(trust, community,config)
        print('dgi_path',self.dgi_path)



        self.prepareModel(config)
        self.temperature_hsic = config['temperature_hsic']
        self.temperature_noise = config['temperature_noise']
        self.lam_t_trust = config['lam_t_trust']
        self.MyDataLoader = MyData(train, trust, feihaoyou, feihaoyou_shequnei, feiling_shequnei, feiling_shequwai,config['seed'], num_ng=1,is_training=True)
        train_loader = self.MyDataLoader
        train_loader.neg_sample()
        userShuffleList = np.random.permutation(self.userNum)

        # print('len(user)', len(user))
        # user_numpu = np.array(user.cpu()).astype(int)
        self.uu_train = train_loader.getTrainInstance(userShuffleList).astype(int)
        # 创建一个布尔掩码，标记每一行是否包含大于的元素

        mask = np.all(self.uu_train <= 7374, axis=1)

        # 使用掩码过滤 self.uu_train
        filtered_uu_train = self.uu_train[mask]

        # 如果需要，可以将 filtered_uu_train 赋值给 self.uu_train
        self.uu_train = filtered_uu_train
    def prepareModel(self,config):
        self.setRandomSeed(config['seed'])
        # one-hot feature
        self.user_feat_sp_tensor = generate_sp_ont_hot(self.userNum).cuda()
        self.dgi = DGI(self.social_graph, self.userNum, config['dgi_hide_dim'], nn.PReLU()).cuda()
        self.dgi.load_state_dict(t.load(self.dgi_path))
        log("load dgi model %s" % (self.dgi_path))
        self.user_dgi_feat = self.dgi.encoder(self.user_feat_sp_tensor).detach()
        if config['dgi_norm'] == 1:
            self.user_dgi_feat = F.normalize(self.user_dgi_feat, p=2, dim=1)
        self.out_dim = sum(eval(config['layer_embedingsize']))

        # self.w_t = nn.Sequential(
        #     nn.Linear(self.out_dim, self.out_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.out_dim, 1, bias=False)
        # ).cuda()
    def gumbel_softmax(self, logits, temperature, hard):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
          logits: [batch_size, n_class] unnormalized log-probs
          temperature: non-negative scalar
          hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
          [batch_size, n_class] sample from the Gumbel-Softmax distribution.
          If hard=True, then the returned sample will be one-hot, otherwise it will
          be a probabilitiy distribution that sums to 1 across classes
        """
        y = self.gumbel_softmax_sample(logits, temperature) ## (0.6, 0.2, 0.1,..., 0.11)
        if hard:
            # k = logits.size(1) # k is numb of classes
            # y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)  ## (1, 0, 0, ..., 0)
            y_hard = torch.eq(y, torch.max(y, dim=1, keepdim=True)[0]).type_as(y)
            y = (y_hard - y).detach() + y
        return y

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        noise = self.sample_gumbel(logits)
        y = (logits + noise) / temperature
        return F.softmax(y, dim=1)

    def sample_gumbel(self, logits):
        """Sample from Gumbel(0, 1)"""
        noise = torch.rand(logits.size())
        eps = 1e-20
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        return torch.Tensor(noise.float()).to(logits.device)
    def forward(self, perturbed=False):
        user_dgi_feat = self.user_dgi_feat
        all_embs = self.get_ego_embeddings(user_dgi_feat)
        embeddings_list = []

        for layer_idx in range(self.n_layers):
            all_embs = self.gcn_conv(all_embs, self.edge_index, self.edge_weight)
            if perturbed:
                random_noise = torch.rand_like(all_embs, device=all_embs.device)


                eps = 1e-20
                random_noise.add_(eps).log_().neg_()
                random_noise.add_(eps).log_().neg_()
                # random_noise =  torch.Tensor(random_noise.float()).to(all_embs.device)

                all_embs = all_embs + torch.sign(all_embs) * F.normalize(random_noise, dim=-1) * self.eps
            embeddings_list.append(all_embs)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def calculate_cl_loss(self, x1, x2):
        x1, x2 = F.normalize(x1, dim=-1), F.normalize(x2, dim=-1)
        pos_score = (x1 * x2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temperature)
        ttl_score = torch.matmul(x1, x2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temperature).sum(dim=1)
        return -torch.log(pos_score / ttl_score).sum()


    def gaussian_kernel(self, X, Y):
        self.sigma = 0.02
        pairwise_distance = torch.cdist(X, Y, p=2).to(torch.float16)
        K = torch.exp(-pairwise_distance.pow(2) / (2 * self.sigma**2)).to(torch.float16)
        return K
    def polynomial_kernel(self,X, Y, degree =10, bias=1.0):
        K = (torch.matmul(X, Y.t()) + bias).pow(degree)
        return K

    def laplacian_kernel(self, X, Y, sigma=0.01):
        pairwise_distance = torch.cdist(X, Y, p=2).to(torch.float16)
        K = torch.exp(-pairwise_distance / sigma)
        return K
    def sigmoid_kernel(self,X, Y, alpha=0.001, bias=0):
        K = torch.sigmoid(alpha * torch.matmul(X, Y.t()).to(torch.float16) + bias).to(torch.float16)
        return K

    def calculate_dgi_loss(self, feat, shuf_feat, dgi_weight):
        positive = self.encoder(feat, corrupt=False)
        negative = self.encoder(shuf_feat, corrupt=True)
        summary = torch.sigmoid(torch.sum(positive*dgi_weight, dim=0))
        positive_total_node_all = self.discriminator(positive, summary)
        negative_total_node_all = self.discriminator(negative, summary)
        l5 = self.loss(positive_total_node_all, torch.ones_like(positive_total_node_all))
        l6 = self.loss(negative_total_node_all, torch.zeros_like(negative_total_node_all))
        return l5+l6


    def predict_loss(self, user_e, pos_e):
        pos_item_score = t.mul(user_e, pos_e).sum(dim=1)
        return pos_item_score
    def calculate_loss(self, interaction):
        loss = super().calculate_loss(interaction)

        user = torch.unique(interaction[self.USER_ID])
        pos_item = torch.unique(interaction[self.ITEM_ID])
        # train_loader = self.MyDataLoader
        # train_loader.neg_sample()
        # user_ori, item_ori = self.forward(perturbed=False)
        # userShuffleList = np.random.permutation(self.userNum)

        # print('len(user)',len(user))
        # user_numpu =np.array(user.cpu()).astype(int)
        # uu_train = train_loader.getTrainInstance(userShuffleList).astype(int)
        # print('uu_train',uu_train)
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
        # 修改好友关系重建的过程

        #     *****************************************************************************************

        user_ori, item_ori = self.forward(perturbed=False)

        perturbed_user_embs_1, perturbed_item_embs_1 = self.forward(perturbed=True)
        perturbed_user_embs_2, perturbed_item_embs_2 = self.forward(perturbed=True)

        user_cl_loss = self.calculate_cl_loss(perturbed_user_embs_1[user], perturbed_user_embs_2[user])
        item_cl_loss = self.calculate_cl_loss(perturbed_item_embs_1[pos_item], perturbed_item_embs_2[pos_item])
        loss_hsic  =  self.hSICLoss(perturbed_user_embs_1[user],perturbed_user_embs_2[user],self.temperature_hsic, hard=False)
        loss_hsic_item  =  self.hSICLoss(perturbed_item_embs_1[pos_item],perturbed_item_embs_2[pos_item],self.temperature_hsic, hard=False)



        loss_reonstruct = 0
        if self.lam_t_trust != 0:
            trust_uid = self.uu_train[:, 0]
            trust_tid = self.uu_train[:, 1]
            trust_neg_uid = self.uu_train[:, 2]
            trust_neg_uid_shequnei = self.uu_train[:, 3]
            # print('trust_uidlen,.', len(trust_uid))
            # print('trust_uid', perturbed_user_embs_1[trust_uid].shape)
            # print('trust_tid', perturbed_user_embs_1[trust_tid].shape)
            # print('trust_neg_uid', trust_neg_uid)
            # print('trust_neg_uid_shequnei', trust_neg_uid_shequnei)

            reconstruct_pos_t = self.predict_loss(user_ori[trust_uid], user_ori[trust_tid])
            reconstruct_neg_t = self.predict_loss(user_ori[trust_uid], user_ori[trust_neg_uid_shequnei])

            reconstruct_neg_t_shequwai = self.predict_loss(user_ori[trust_uid], user_ori[trust_neg_uid])
            self.trust_reconstruct_loss = 0.2 * (
                - ((reconstruct_pos_t.view(-1) - reconstruct_neg_t.view(-1)).sigmoid()+10e-8).log().sum()) + 0.8 * (- ((
                                                                                                                         reconstruct_neg_t.view(
                                                                                                                             -1) - reconstruct_neg_t_shequwai.view(
                                                                                                                     -1)).sigmoid()+10e-8).log().sum())
            # self.trust_reconstruct_loss = 0.8 * (
            #     - (((reconstruct_pos_t.view(-1) - reconstruct_neg_t.view(-1)).sigmoid())+10e-8).log().sum())
        if self.lam_t_trust != 0:
            loss_reonstruct = ((self.trust_reconstruct_loss * self.lam_t_trust) / self.uu_train.shape[0])
        return loss + self.cl_rate * (user_cl_loss + item_cl_loss)+self.hsic * (loss_hsic+loss_hsic_item) + loss_reonstruct
        # return loss + self.cl_rate * (user_cl_loss + item_cl_loss)+self.hsic *(loss_hsic) +loss_reonstruct
        # return loss + self.cl_rate * (user_cl_loss + item_cl_loss)+self.hsic *(loss_hsic_item) +loss_reonstruct

        # return loss + self.cl_rate * (user_cl_loss + item_cl_loss)+self.hsic *(loss_hsic+loss_hsic_item)


        # return loss + self.cl_rate * (user_cl_loss + item_cl_loss)+self.hsic *(loss_hsic)

        # return loss + self.cl_rate * (user_cl_loss + item_cl_loss)



    def preTrain(self, trust, community,config):
        # 下面是原文的，从节点层级到全局图层级的嵌入学习过程，用到了dgi库

        # 先是图的统计资料
        # print(trust.s)
        # print(trust.T)
        tmpMat = (trust + trust.T)
        self.userNum = trust.shape[0]
        # userNum, itemNum = train.shape
        adj = (tmpMat != 0) * 1
        adj = adj + sp.eye(adj.shape[0])

        adj = adj.tocsr()  # 生成邻接矩阵

        nodeDegree = np.sum(adj, axis=1)  # 生成节点度
        degreeSum = np.sum(nodeDegree)  # 度总数
        dgi_weight = t.from_numpy((nodeDegree + 1e-6) / degreeSum).float().cuda()

        # 社区的统计资料
        adj_community = (community != 0) * 1
        adj_community = adj_community.tocsr()

        community_Degree = np.sum(adj_community, axis=1)  # 生成用户链接社区数
        community_DegreeSum = np.sum(community_Degree)  # 生成社区总数
        dgi_weight_community = t.from_numpy((community_Degree + 1e-6) / community_DegreeSum).float().cuda()

        user_feat_sp_tensor = generate_sp_ont_hot(self.userNum ).cuda()  # 生成独热用户向量，把用户id转化成一个独热码
        in_feats =  self.userNum

        edge_src, edge_dst = adj.nonzero()  # 根据边（a,b）生成图，这里的adj是矩阵
        self.social_graph = dgl.graph(data=(edge_src, edge_dst),
                                      idtype=t.int32,
                                      num_nodes=trust.shape[0] ,
                                      device=device_gpu)

        # # # DGI是图表示方法，主要依赖于 最大限度地扩大图增强表示和目前提取到的图信息之间的互信息
        dgi = DGI(self.social_graph, in_feats, config['dgi_hide_dim'], nn.PReLU()).cuda()
        dgi_optimizer = t.optim.Adam(dgi.parameters(), lr=config['dgi_lr'], weight_decay=config['dgi_weigh_decay'])

        # dgi = DGI(self.social_graph, in_feats, args.dgi_hide_dim, nn.LeakyReLU()).cuda()
        # dgi_optimizer = t.optim.Adam(dgi.parameters(), lr=args.dgi_lr, weight_decay=args.dgi_reg)
        cnt_wait = 0
        best = 1e9
        best_t = 0
        for epoch in range(500):
            # 下面这个是对原始图
            dgi.train()
            dgi_optimizer.zero_grad()
            idx = np.random.permutation(self.userNum )
            shuf_feat = sparse_mx_to_torch_sparse_tensor(sp.eye(self.userNum ).tocsr()[idx]).cuda()
            idx_shequ = np.random.permutation(self.userNum)
            shuf_feat_shequ = sparse_mx_to_torch_sparse_tensor(sp.eye(self.userNum).tocsr()[idx_shequ]).cuda()
            # 稀疏矩阵转化为torch中的稀疏矩阵
            loss = dgi(user_feat_sp_tensor, shuf_feat, dgi_weight, dgi_weight_community, shuf_feat_shequ)
            loss.backward()
            dgi_optimizer.step()
            # if loss+loss_COM < best:
            if loss < best:
                best = loss
                cnt_wait = 0
                DIR = '/home/nixuelian/code_project/RecBole-GNN-main/saved'
                # DIR = os.path.join(os.getcwd(), "Model", self.args.dataset)

                # path = DIR + r"/dgi_" + modelUTCStr + "_" + 'yelp_80w' + "_" + "_" + str(
                #     config['dgi_hide_dim']) + "_" + str(config['dgi_weigh_decay'])
                # path += '.pth'
                path = DIR + r"/dgi_" + modelUTCStr + "_" + 'ciao' + "_" + "_" + str(
                    config['dgi_hide_dim']) + "_" + str(config['dgi_weigh_decay'])
                path += '.pth'
                t.save(dgi.state_dict(), path)
            else:
                cnt_wait += 1

            if cnt_wait == 5:
                print('DGI Early stopping!')
                # print(path)
                return path
            # 上面就是为了完成预训练的过程。根据输入id，生成合适的独热编码
class HSICLoss(GeneralGraphRecommender):
    # def __init__(self, sigma):
    #     super(HSICLoss, self).__init__()

    def __init__(self, config, dataset):
        super(HSICLoss, self).__init__(config, dataset)
        # self.sigma = 0.01
        self.hsic_xx_lamb = config['hsic_xx_lamb']
        self.hsic_yy_lamb = config['hsic_yy_lamb']

        self.sigma = config['sigma']
        print('self.sigma',self.sigma)
    def gaussian_kernel(self, X, Y):
        pairwise_distance = torch.cdist(X, Y, p=2).to(torch.float16)
        K = torch.exp(-pairwise_distance.pow(2) / (2 * self.sigma**2)).to(torch.float16)
        return K

    def linear_kernel(self, X, Y):
        K = torch.matmul(X, Y.t())
        return K

    def polynomial_kernel(self,X, Y, degree =1, bias=0):
        K = (torch.matmul(X, Y.t()) + bias).pow(degree)
        return K

    def laplacian_kernel(self, X, Y, sigma=0.01):
        pairwise_distance = torch.cdist(X, Y, p=2).to(torch.float16)
        K = torch.exp(-pairwise_distance / sigma)
        return K
    def sigmoid_kernel(self,X, Y, alpha=1, bias=0):
        K = torch.sigmoid(alpha * torch.matmul(X, Y.t()).to(torch.float16) + bias).to(torch.float16)
        return K
    def hsic_loss(self, X, Y,temperature,hard):

        # 都归一化
        X, Y = F.normalize(X, dim=-1), F.normalize(Y, dim=-1)
        # K_X_weighted = K_X * feature_weights.unsqueeze(1) * feature_weights.unsqueeze(0)

        # 权重
        feature_X_weights =self.gumbel_softmax(X, temperature, hard)
        # feature_Y_weights =self.gumbel_softmax(Y, temperature, hard)

        # X = X*feature_X_weights.unsqueeze(1) * feature_X_weights.unsqueeze(0)
        # Y = Y*feature_Y_weights.unsqueeze(1) *feature_Y_weights.unsqueeze(0)

        X = X*feature_X_weights
        # Y = Y*feature_Y_weights

        # X = feature_X_weights
        # # Y = feature_Y_weights
        # X, Y = F.normalize(X, dim=-1), F.normalize(Y, dim=-1)
        Kx = self.gaussian_kernel(X, X).to(torch.float16)
        Ky = self.gaussian_kernel(Y, Y).to(torch.float16)
        # Kx = Kx*feature_X_weights
        # Ky = Ky*feature_Y_weights

        # X, Y = F.normalize(X, dim=-1), F.normalize(Y, dim=-1)
        n = X.size(0)
        H = (torch.eye(n) - torch.ones(n, n) / n).to(torch.float16).cuda()
        Kx_centered = torch.matmul(torch.matmul(H, Kx).to(torch.float16), H).to(torch.float16).cuda()
        Ky_centered = torch.matmul(torch.matmul(H, Ky).to(torch.float16), H).to(torch.float16).cuda()
        hsic = torch.trace(torch.matmul(Kx_centered, Ky_centered)).to(torch.float16).cuda()
        return hsic

    def gumbel_softmax(self, logits, temperature, hard):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
          logits: [batch_size, n_class] unnormalized log-probs
          temperature: non-negative scalar
          hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
          [batch_size, n_class] sample from the Gumbel-Softmax distribution.
          If hard=True, then the returned sample will be one-hot, otherwise it will
          be a probabilitiy distribution that sums to 1 across classes
        """
        y = self.gumbel_softmax_sample(logits, temperature) ## (0.6, 0.2, 0.1,..., 0.11)
        if hard:
            # k = logits.size(1) # k is numb of classes
            # y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)  ## (1, 0, 0, ..., 0)
            y_hard = torch.eq(y, torch.max(y, dim=1, keepdim=True)[0]).type_as(y)
            y = (y_hard - y).detach() + y
        return y

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        noise = self.sample_gumbel(logits)
        y = (logits + noise) / temperature
        return F.softmax(y, dim=1)

    def sample_gumbel(self, logits):
        """Sample from Gumbel(0, 1)"""
        noise = torch.rand(logits.size())
        eps = 1e-20
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        return torch.Tensor(noise.float()).to(logits.device)

    def forward(self, X, Y,temperature,hard):


        loss = -self.hsic_loss(X, Y,temperature,hard).cuda().to(torch.float16)+self.hsic_xx_lamb*torch.sqrt(self.hsic_loss(X, X,temperature,hard).cuda()).to(torch.float16)+self.hsic_yy_lamb*torch.sqrt(self.hsic_loss(Y, Y,temperature,hard).cuda()).to(torch.float16)
        # print('loss函数里',loss)

        #
        # loss = -self.hsic_loss(X, Y,temperature,hard).cuda().to(torch.float16)+self.hsic_xx_lamb*torch.sqrt(self.hsic_loss(X, X,temperature,hard).cuda()).to(torch.float16)
        #     loss = -self.hsic_loss(X, Y).cuda()+0.05*torch.sqrt(self.hsic_loss(X, X).cuda())


        return loss






import random
from collections import defaultdict

class MyData():
    def __init__(self, trainMat, trustMat, feihaoyouMat,feihaoyouMat_shequnei,feiling_shequnei,feiling_shequwai, seed, num_ng=0, is_training=None):
        super(MyData, self).__init__()
        self.setRandomSeed(seed)
        self.trainMat  = trainMat
        self.trustMat = trustMat
        # self.userNum, self.itemNum = trainMat.shape
        self.userNum = 7374
        self.num_ng = num_ng
        self.is_training = is_training
        self.feihaoyouMat = feihaoyouMat
        self.feihaoyouMat_shequnei = feihaoyouMat_shequnei
        self.feiling_shequnei = feiling_shequnei
        self.feiling_shequwai=feiling_shequwai

        train_u1, train_u2 = self.trustMat[:self.userNum ,:self.userNum ].nonzero()

        self.trust_data = np.hstack(
            (train_u1.reshape(-1, 1), train_u2.reshape(-1, 1)))
        self.trust_data = self.trust_data.astype(np.int_)



        length = self.userNum

        self.feiling_col_shequnei=np.random.randint(low=0, high=self.userNum, size=length)
        for i in range(length):
            self.feiling_col_shequnei[i] = np.random.randint(low=1, high=self.feiling_shequnei[i, 0]-1, size=1)


        self.feiling_col_shequwai=np.random.randint(low=0, high=self.userNum, size=length)
        for i in range(length):
            self.feiling_col_shequwai[i] = np.random.randint(low=1, high=self.feiling_shequwai[i, 0]-1, size=1)
            # 生成随机整数数组 self.feiling1

        # self.feiling_col_shequnei = np.random.randint(low=0, high=self.userNum, size=length)
        #
        #     # 生成随机整数数组 self.feiling1 中每个元素的取值
        # self.feiling_col_shequnei[:, 0] = np.random.randint(low=1, high=self.feiling_shequnei[:, 0] - 1, size=length)
        # self.feiling_col_shequwai = np.random.randint(low=0, high=self.userNum, size=length)
        #
        #     # 生成随机整数数组 self.feiling1 中每个元素的取值
        # self.feiling_col_shequwai[:, 0] = np.random.randint(low=1, high=self.feiling_shequwai[:, 0] - 1, size=length)

            # print(self.feiling_shequwai[i, 0])
        # print(self.feiling_col_shequwai)

        # adj_DIR = os.path.join(os.getcwd(), "data", dataset, 'mats')
        # adj_path = adj_DIR + '/{0}_multi_item_adj.pkl'.format(args.rate)
        # adj_path_2 = adj_DIR + '/{0}_feihaoyou.pkl'.format(args.rate)
        # adj_path_3 = adj_DIR + '/{0}_feihaoyou_shequnei.pkl'.format(args.rate)
        # adj_path_4 = adj_DIR + '/{0}_community.pkl'.format(args.rate)
        #
        # with open(adj_path, 'rb') as fs:
        #     multi_adj = pickle.load(fs)


        # feihaoyou_u, feihaoyou_u2 = self.feihaoyouMat.nonzero()
        # self.feihaoyou_data = np.hstack(
        #     (feihaoyou_u.reshape(-1, 1), feihaoyou_u2.reshape(-1, 1)))
        # self.feihaoyou_data = self.feihaoyou_data.astype(np.int)
        # assert np.sum(self.feihaoyou_u.data == 0) == 0
        # self.feihaoyou_u = np.hstack(
        #     (train_u.reshape(-1, 1), train_v.reshape(-1, 1), train_r.reshape(-1, 1)))
        # self.feihaoyou_u = self.feihaoyou_u.astype(np.int)

    def setRandomSeed(self, seed):
        np.random.seed(seed)
        t.manual_seed(seed)
        t.cuda.manual_seed(seed)
        random.seed(seed)

    def neg_sample(self):
        # self.train_neg_sample()
        self.trust_neg_sample()

    # 这是修改后生成负的好友样本
    def trust_neg_sample(self):
        # assert self.is_training
        # tmp_trustMat = self.trustMat.todok()
        length = self.trust_data.shape[0]
        # trust_neg_data = np.random.randint(low=0, high=self.userNum, size=length)
        self.trust_data_dict = defaultdict(list)
        self.shequ_zongshu = 20

        for i in range(length):
            uid = self.trust_data[i][0]
            uid_trust = self.trust_data[i][1]
            # neg_id_feihaoyou =self.feihaoyouMat[i]

            # non_zero_col_feihaoyou = np.nonzero(self.feihaoyouMat[i])[0]
            # selected_col = np.random.choice(non_zero_col_feihaoyou)
            # selected_value = self.feihaoyouMat[i, selected_col]
            #
            # non_zero_col_feihaoyou_shequnei = np.nonzero(self.feihaoyouMat_shequnei[i])[0]
            # selected_col_shequnei = np.random.choice(non_zero_col_feihaoyou_shequnei)
            # selected_value_shequnei = self.feihaoyouMat_shequnei[i, selected_col_shequnei]


            selected_value = self.feihaoyouMat[uid, self.feiling_col_shequwai[uid]]

            # non_zero_col_feihaoyou_shequnei = np.nonzero(self.feihaoyouMat_shequnei[i])[0]
            # selected_col_shequnei = np.random.choice(non_zero_col_feihaoyou_shequnei)
            selected_value_shequnei = self.feihaoyouMat_shequnei[uid, self.feiling_col_shequnei[uid]]



            # neg_fid = trust_neg_data[i]
            # if (uid, neg_fid) in tmp_trustMat:
            #     while (uid, neg_fid) in tmp_trustMat:
            #         neg_fid = np.random.randint(low=0, high=self.userNum)
            #     trust_neg_data[i] = neg_fid

            self.trust_data_dict[uid].append([uid, self.trust_data[i][1], selected_value,selected_value_shequnei])


    def train_neg_sample(self):
        #'no need to sampling when testing'
        assert self.is_training
        self.train_data_dict = defaultdict(list)

        length = self.trainMat.data.size
        train_data = self.trainMat.data
        train_neg_data = np.random.randint(low=1, high=self.ratingClass+1, size=length)

        rebuild_idx = np.where(train_data == train_neg_data)[0]

        for idx in rebuild_idx:
            val = np.random.randint(1, self.ratingClass+1)
            while val == train_data[idx]:
                val = np.random.randint(1, self.ratingClass+1)
            train_neg_data[idx] = val

        assert np.sum(train_data == train_neg_data) == 0


        neg_item_id = np.random.randint(low=0, high=self.trainMat.shape[1], size=length)
        for i in range(length):
            uid = self.train_data[i][0]
            iid = self.train_data[i][1]
            rating = self.train_data[i][2]
            neg_rating = train_neg_data[i]
            while self.trainMat[uid, neg_item_id[i]] != 0:
                neg_item_id[i] =  np.random.randint(low=0, high=self.trainMat.shape[1], size=1)
            # neg_item_ids[i] = neg_item_id
            neg_rating_2 =neg_item_id[i]
            self.train_data_dict[uid].append([uid, iid, rating, neg_rating,neg_rating_2])



    def getTrainInstance(self, userIdxList):
        # ui_train = []
        uu_train = []

        for uidx in userIdxList:
            # ui_train += self.train_data_dict[uidx]
            uu_train += self.trust_data_dict[uidx]

        # ui_train = np.array(ui_train)
        # idx = np.random.permutation(len(ui_train))
        # ui_train = ui_train[idx]

        uu_train = np.array(uu_train)
        idx = np.random.permutation(len(uu_train))
        uu_train = uu_train[idx]

        return uu_train
