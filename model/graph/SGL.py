import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from data.augmentor import GraphAugmentor

# Paper: self-supervised graph learning for recommendation. SIGIR'21

# 看代码顺序
# 从 train(self)逐渐往下看



class SGL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SGL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['SGL'])
        self.cl_rate = float(args['-lambda'])
        aug_type = self.aug_type = int(args['-augtype'])
        drop_rate = float(args['-droprate'])
        n_layers = int(args['-n_layer'])
        temp = float(args['-temp'])
        self.model = SGL_Encoder(self.data, self.emb_size, drop_rate, n_layers, temp, aug_type)

    # 训练
    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            dropped_adj1 = model.graph_reconstruction()  # 得到破坏后的图 G1
            dropped_adj2 = model.graph_reconstruction()  # 得到破坏后的图 G2
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()  # 进入forword函数。在该函数中，会利用图G得到用户和物品的嵌入向量。
                # 是否存在疑问，利用G1和G2得到嵌入向量的过程在哪里？它在计算对比损失的函数中。
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]  # 从总体嵌入中抽取自己需要的嵌入向量（抽取部分用户，部分正对物品，部分负对物品）
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)  # 计算BPR损失
                cl_loss = self.cl_rate * model.cal_cl_loss([user_idx,pos_idx],dropped_adj1,dropped_adj2)  # 计算对比损失
                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb,neg_item_emb) + cl_loss  # BPR损失 + L2正则化 + 对比损失
                # =====对模型进行反向传播，优化参数======
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                # ==================================
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            if epoch>=5:
                self.fast_evaluation(epoch)  # 对模型进行评估
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class SGL_Encoder(nn.Module):
    def __init__(self, data, emb_size, drop_rate, n_layers, temp, aug_type):
        super(SGL_Encoder, self).__init__()
        self.data = data
        self.drop_rate = drop_rate
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.temp = temp
        self.aug_type = aug_type
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    # 如何得到破坏图
    def graph_reconstruction(self):
        if self.aug_type==0 or 1:
            dropped_adj = self.random_graph_augment()  # 丢弃节点或者丢弃边
        else:
            dropped_adj = []
            for k in range(self.n_layers):
                dropped_adj.append(self.random_graph_augment())
        return dropped_adj

    def random_graph_augment(self):
        dropped_mat = None
        if self.aug_type == 0:
            dropped_mat = GraphAugmentor.node_dropout(self.data.interaction_mat, self.drop_rate)
        elif self.aug_type == 1 or self.aug_type == 2:
            dropped_mat = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)  # 看这个，按照特定比例丢弃边。在data下的augmentor.py中
        dropped_mat = self.data.convert_to_laplacian_mat(dropped_mat)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda()

    # forword函数。这里面都是LightGCN模型学习嵌入的过程。
    def forward(self, perturbed_adj=None):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)  # 在竖直方向上拼接向量
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):  # layers 训练轮次
            if perturbed_adj is not None:  # 是否使用图结构被破坏的图进行训练。默认为否，即默认使用图G进行训练
                if isinstance(perturbed_adj,list):
                    ego_embeddings = torch.sparse.mm(perturbed_adj[k], ego_embeddings)  # 这个不用看
                else:
                    ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings) # 使用图G1或者G2进行训练。使用哪个看你传了哪个。后续看对比学习部分你就知道了。
            else:
                ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)  # 使用图G进行训练
            all_embeddings.append(ego_embeddings)  # 将所有训练得到的嵌入都记录到列表中

        all_embeddings = torch.stack(all_embeddings, dim=1)  # 这个你不看
        all_embeddings = torch.mean(all_embeddings, dim=1)  # 将每轮训练得到的嵌入进行相加后求平均 
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])  # 从总的嵌入中得到用户和物品嵌入
        return user_all_embeddings, item_all_embeddings

    # 对比学习部分
    def cal_cl_loss(self, idx, perturbed_mat1, perturbed_mat2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.forward(perturbed_mat1)  # 通过图 G1 得到嵌入（用户和物品）
        user_view_2, item_view_2 = self.forward(perturbed_mat2)  # 通过图 G2 得到嵌入（用户和物品）
        view1 = torch.cat((user_view_1[u_idx],item_view_1[i_idx]),0)  # 将通过G1得到的嵌入在竖直方向上进行拼接。用户的在上面，物品的在下面
        view2 = torch.cat((user_view_2[u_idx],item_view_2[i_idx]),0)  # 将通过G1得到的嵌入在竖直方向上进行拼接。用户的在上面，物品的在下面
        return InfoNCE(view1,view2,self.temp)  # 计算对比损失，找到这个函数。它在util下的 loss_torch.py中

