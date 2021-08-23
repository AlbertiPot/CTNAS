import torch
import torch.nn as nn

from .gcn import GCN

from core.dataset.tensorize import nasbench_tensor2arch
from core.utils import _sign
from core.config import args


class NAC(nn.Module):
    def __init__(self, n_nodes, n_ops, n_layers=2, ratio=2, embedding_dim=128):
        super(NAC, self).__init__()
        self.n_nodes = n_nodes
        self.n_ops = n_ops

        self.embedding = nn.Embedding(self.n_ops+3, embedding_dim=embedding_dim)                            # 采用torch.nn.Embedding对输入的算子进行编码
        
        self.gcn = GCN(n_layers=n_layers, in_features=embedding_dim,
                       hidden=embedding_dim, num_classes=embedding_dim)
        
        self.fc = nn.Linear(embedding_dim*ratio, 1, bias=True)

    def forward(self, arch0, arch1):

        feature = torch.cat([self.extract_features(arch0), self.extract_features(arch1)], dim=1)            # 拼接baseline和被比较arch的feature，函数extract_features提取邻接矩阵和算子编码的feature map

        score = self.fc(feature).view(-1)                                                                   # 嵌入输入fc处理得到

        probility = torch.sigmoid(score)                                                                    # 输入logit，得到概率

        return probility

    def extract_features(self, arch):
        if len(arch) == 2:                                                                                  # 如果arch仅仅包含了一个结构的算子和邻接矩阵
            matrix, op = arch
            return self._extract(matrix, op)
        else:                                                                                               # arch若包含了两个结构的算子和邻接矩阵，拆开后分别调用
            nor_mat, nor_ops, red_mat, red_ops = arch
            nor_feature = self._extract(nor_mat, nor_ops)
            red_feature = self._extract(red_mat, red_ops)
            return torch.cat([nor_feature, red_feature], dim=1)

    def _extract(self, matrix, ops):                                                                        # 输入是[batchsize=256,nodes_ops=7] 256的batchsize，共7个节点，每个节点上有3个算子的，但是本文的embedding设置为了n_ops+3=6个词
        ops = self.embedding(ops)                                                                           # 调用nn.Embedding处理算子的编码，输出[256,7,128],共256个结构，每个节点上的算子以128位长的数字编码
        feature = self.gcn(ops, matrix).mean(dim=1, keepdim=False)                                          # gcn(ops=[256,7,128],matrix = [256,7,7])处理得到feature[256,128]
        return feature


def linear_bn_relu(in_features, out_features, relu=True):
    layers = [nn.Linear(in_features, out_features)]
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return layers


class MLP(nn.Sequential):
    def __init__(self, in_features, hidden_features, out_features, n_layers):
        assert n_layers > 1
        layers = []
        for i in range(n_layers):
            in_f = in_features if i == 0 else hidden_features
            out_f = out_features if i == n_layers-1 else hidden_features
            relu = i != n_layers-1
            layers += linear_bn_relu(in_f, out_f, relu)
        super(MLP, self).__init__(*layers)


class MBSpaceNAC(nn.Module):
    def __init__(self, in_features, hidden_features, n_layers=3):
        super(MBSpaceNAC, self).__init__()

        self.linear_extractor = MLP(in_features, hidden_features, hidden_features, n_layers)            # 以MLP作为encoder，即线型编码提取器，提取特征

        self.fc = nn.Linear(hidden_features*2, 1, bias=True)

    def forward(self, arch0, arch1):
        if isinstance(arch0, list):
            arch0 = arch0[0]
        if isinstance(arch1, list):
            arch1 = arch1[0]
        # import ipdb; ipdb.set_trace()
        feature = torch.cat([self.linear_extractor(arch0), self.linear_extractor(arch1)], dim=1)
        score = self.fc(feature).view(-1)
        probility = torch.sigmoid(score)

        return probility
