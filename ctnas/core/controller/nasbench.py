import os
import numpy as np
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F


class NASBenchController(nn.Module):
    def __init__(self, n_ops=3, n_nodes=5, hidden_size=64,
                 temperature=None,  tanh_constant=1.5, op_tanh_reduce=2.5, device="cpu"):
        super(NASBenchController, self).__init__()
        self.n_ops = n_ops
        self.n_nodes = n_nodes
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.tanh_constant = tanh_constant
        self.op_tanh_reduce = op_tanh_reduce

        self.n_nodes_embedding = nn.Embedding(n_nodes, self.hidden_size)
        self.node_embedding = nn.Embedding(n_nodes + 2, self.hidden_size)
        self.op_embedding = nn.Embedding(n_ops, self.hidden_size)

        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.emb_attn = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.hid_attn = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_attn = nn.Linear(self.hidden_size, 1, bias=False)
        self.w_soft = nn.Linear(self.hidden_size, self.n_ops)

        # classify how many nodes
        self.n_soft = nn.Linear(self.hidden_size, self.n_nodes)

        self.device = device
        self.batch_size = 1

        self.reset_parameters()

    def reset_parameters(self, init_range=0.1):
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)

    @functools.lru_cache(maxsize=128)
    def _zeros(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size), device=self.device, requires_grad=False)

    def _scale_attention(self, logits, temperature, tanh_constant, constant_reduce=None):
        if temperature is not None:
            logits /= temperature
        if tanh_constant is not None:
            if constant_reduce is not None:
                tanh_constant /= constant_reduce
            logits = tanh_constant * torch.tanh(logits)
        return logits

    def _impl(self, probs):
        m = torch.distributions.Categorical(probs=probs)                                # 从0 → K-1的整数中采样， K是概率列表的长度.
        action = m.sample().view(-1)
        select_log_p = m.log_prob(action)
        entropy = m.entropy()
        return action, select_log_p, entropy

    def forward(self, force_uniform=False):
        arch_seq = []
        logp_buf = []
        entropy_buf = []

        inputs = self._zeros(self.batch_size)
        hidden = self._zeros(self.batch_size), self._zeros(self.batch_size)         # hidden是隐层变量，即隐层特征hx和内部状态cx

        # 选择节点的数量
        if force_uniform:
            logits = torch.zeros(self.n_nodes, device=self.device)
        else:
            embed = inputs
            hx, cx = self.lstm(embed, hidden)
            hidden = (hx, cx)
            logits = self.n_soft(hx).view(-1)                                       # 输出维度是节点的数量
            logits = self._scale_attention(logits, self.temperature, self.tanh_constant, self.op_tanh_reduce)
        probs = F.softmax(logits, dim=-1)
        action, select_log_p, entropy = self._impl(probs)                           # 从0到K-1个节点的对应概率中采样一个节点数（注意是实际节点数-1）

        arch_n_nodes = action.item() + 1                                            # action -> how many nodes，+1是因为action对应的是节点数比实际节点数少一
        arch_seq.append(arch_n_nodes)
        logp_buf.append(select_log_p)
        entropy_buf.append(entropy)

        inputs = action
        embed = self.n_nodes_embedding(inputs)

        """
        Here, we make decisions for the op type and previous node id of each node
        上一步获得节点数后，计算每个节点的
        1)前一个节点的ID和
        2)节点的算子类型
        """
        for node_idx in range(arch_n_nodes+1):
            # first, choice for previous node is
            if force_uniform:
                logits = torch.zeros(node_idx+1, device=self.device)
            else:
                hx, cx = self.lstm(embed, hidden)                                   # 这一次的embed是上一歩的节点数目的编码，hidden是上一个lstm的输出的hidden=(hx,cx)
                hidden = (hx, cx)
                query = self.node_embedding.weight[:node_idx+1, :]                  # 取当前id节点前全部节点的编码
                query = torch.tanh(self.emb_attn(query) + self.hid_attn(hx))        # tanh激活（嵌入的全连接注意力)+隐层状态的全连接注意力(hx)，输出query
                logits = self.v_attn(query).view(-1)                                # 对query过一下全连接注意力后输出,编码维度压缩到1（若是第6个节点，则有5*1个前序节点的压缩后编码)，即5个前序节点仅选择1个前序节点
                logits = self._scale_attention(logits, self.temperature, self.tanh_constant)
            probs = F.softmax(logits, dim=-1)
            action, select_log_p, entropy = self._impl(probs)                       # 根据前序节点的概率采样一个前序节点id
            arch_seq.append(action.item())  # action -> previous node is
            logp_buf.append(select_log_p)
            entropy_buf.append(entropy)

            inputs = action
            embed = self.node_embedding(inputs)

            # the output node does not to make decision for op type
            if node_idx == arch_n_nodes:                                            # 循环是为了找每个节点之前的节点，为了和output连接，在循环前半部分的找节点部分要用到output节点，而循环后半部分不需要对输出节点找算子
                break

            # second, choice for node type
            if force_uniform:
                logits = torch.zeros(self.n_ops, device=self.device)
            else:
                hx, cx = self.lstm(embed, hidden)
                hidden = (hx, cx)
                logits = self.w_soft(hx).view(-1)                                   # 输出维度是算子的个数
                logits = self._scale_attention(logits, self.temperature, self.tanh_constant, self.op_tanh_reduce)
            probs = F.softmax(logits, dim=-1)
            action, select_log_p, entropy = self._impl(probs)                       # 根据算子对应的概率采样出算子类型
            arch_seq.append(action.item())  # action -> node type
            logp_buf.append(select_log_p)
            entropy_buf.append(entropy)
            inputs = action
            embed = self.op_embedding(inputs)

        # we have (9-arch_n_nodes-1) edges left, we should make decisions for the start
        # and end of the left edges
        # arch_n_nodes是指中间节点，若=5，加上input/output两个输出节点，一个7个节点共(arch_n_nodes+1)个边，最大9个边，还剩9-(arch_n_nodes+1)个边
        for i in range(2*(9-arch_n_nodes-1)):                                       # 剩余的edge边，找到其开始和结束节点
            if force_uniform:
                logits = torch.zeros(arch_n_nodes+2, device=self.device)
            else:
                hx, cx = self.lstm(embed, hidden)
                hidden = (hx, cx)
                query = self.node_embedding.weight[:arch_n_nodes+2, :]              # ?这里检索嵌入，是不是应该根据采样的节点找对应的索引，而不是找前几个索引加上2个输入输出索引
                query = torch.tanh(self.emb_attn(query) + self.hid_attn(hx))
                logits = self.v_attn(query).view(-1)
                logits = self._scale_attention(logits, self.temperature, self.tanh_constant)
            probs = F.softmax(logits, dim=-1)
            action, select_log_p, entropy = self._impl(probs)                       # 采样的一个节点，奇数序号是开始节点，偶数节点是结尾节点
            arch_seq.append(action.item())  # action -> start or end of the remain edges
            logp_buf.append(select_log_p)
            entropy_buf.append(entropy)

            inputs = action
            embed = self.node_embedding(inputs)

        return arch_seq, sum(logp_buf), sum(entropy_buf)
