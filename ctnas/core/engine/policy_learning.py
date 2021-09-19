
import os
import copy
import random
import functools

from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.utils.tensorboard as tensorboard

from core.model import NAC
from core.metric import AverageMetric, MovingAverageMetric
from core.genotypes import Genotype
from core.dataset.database import DataBase
from core.dataset.architecture.common_nas import PRIMITIVES
from core.dataset.utils import ControllerDataset
from core.dataset.seq2arch import seq2arch_fn
from core.dataset.tensorize import tensorize_fn, nasbench_tensor2arch
from core.controller import NASBenchController
from core.config import args
from core.utils import *


def single_batchify(*items):
    return [item.unsqueeze(0) for item in items]


best_iters = -1


def train_controller(max_iter: int, database: DataBase,
                     entropy_coeff: float, grad_clip: int,
                     controller: NASBenchController, nac: NAC,
                     optimizer: optim.Optimizer, writer: tensorboard.SummaryWriter,
                     alternate_train, alternate_evaluate, random_baseline=False,
                     log_frequence: int = 10, search_space=None):
    controller.train()
    nac.eval()
    optimizer.zero_grad()

    policy_loss_avg = MovingAverageMetric()
    entropy_mavg = MovingAverageMetric()
    logp_mavg = MovingAverageMetric()
    score_avg = MovingAverageMetric()

    pseudo_architecture_set = None

    # 采样baseline结构
    with torch.no_grad():
        *arch_seq, _, _ = controller(force_uniform=True)                                    # 用controller随机生成一个子网, 输出是一个list整合了[节点数, 前一节点id和当前节点的算子类型, 剩余连接的起始节点]:[[2, 0, 1, 0, 2, 0, 2, 0, 2, 1, 3, 0, 1, 3, 2, 3, 2, 2]]
        raw_arch = seq2arch_fn(arch_seq)
        """
        调整
       matrix=array([
       [0, 1, 1, 1],
       [0, 0, 1, 1],
       [0, 0, 0, 1],
       [0, 0, 0, 0]], dtype=int32), 
       ops=['input', 'conv1x1-bn-relu', 'maxpool3x3', 'output'], 
       n_params=None, 
       training_time=None, 
       train_accuracy=None, 
       validation_accuracy=None, 
       test_accuracy=None, 
       hash_=None, 
       rank=None)
        """
        baseline_arch = [tensorize_fn(raw_arch, device=device)]                         # 上三角阵处理为下三角阵，将算子字符变成编号
        """
        [(tensor([
        [0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.]], device='cuda:0'), 
        
        tensor([1, 2, 4, 5, 0, 0, 0], device='cuda:0'))
        ]
        """

    best_collect_archs = [arch_seq]

    for iter_ in range(max_iter):

        if iter_ % args.n_iteration_update_pseudoset == 0 and args.pseudo_ratio != 0:
            
            """
            1. 采用controller生成数据对，根据NAC输出的概率保留topK个作为新的数据集补充
            """
            if pseudo_architecture_set is None:
                pseudo_architecture_set = \
                    generate_architecture_with_pseudo_labels(                           # 返回值是 arch0,arch1,labels
                        nac, controller,
                        2*int(args.pseudo_ratio*args.train_batch_size),                 # 总的遍历结构pair的数目是total, 计算其置信度和标签，这里是2倍的伪数据集率*batch sz
                        int(args.pseudo_ratio*args.train_batch_size))                   # 最后保留的置信度最高的k个结构对，这里k=batch_size
            else:
                pseudo_architecture_set = list_concat(
                    pseudo_architecture_set,
                    generate_architecture_with_pseudo_labels(
                        nac, controller,
                        2*args.n_sample_architectures, args.n_sample_architectures)
                )
            
            """
            2. 采用刚补充的数据集，训练NAC
            """
            epoch = args.nac_epochs + iter_                                             # 之前NAC已经训了1000个eps，这里生成伪数据集后，继续NAC训练一个epoch
            accuracy, rank_loss = alternate_train(epoch=epoch, pseudo_set=pseudo_architecture_set)
            writer.add_scalar("nac/train_accuracy", accuracy, epoch)
            writer.add_scalar("nac/loss", rank_loss, epoch)
            KTau = alternate_evaluate(epoch=epoch)                                      # 评估NAC这个eps训练的结果
            writer.add_scalar("nac/ktau", KTau, epoch)

        """
        3. 采用controller采样一个结构及其对数概率和熵，用这个结构和baseline对比得到一个score，用来计算loss并更新梯度
        """
        *arch_seq, logp, entropy = controller()                                         # 采样一个结构，并且输出这个采样的对数概率和熵
        with torch.no_grad():
            sample_arch = [tensorize_fn(seq2arch_fn(arch_seq), device=device)]
            score = nac(batchify(sample_arch), batchify(baseline_arch))                 # 用nac比较采样结构和baseline的好坏，输出一个分数
            score = score.mean().item()

        policy_loss = -logp * score - entropy_coeff * entropy

        optimizer.zero_grad()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(controller.parameters(), grad_clip)
        policy_loss.backward()
        optimizer.step()

        policy_loss_avg.update(policy_loss)
        entropy_mavg.update(entropy)
        logp_mavg.update(logp)
        score_avg.update(score)

        if iter_ % log_frequence == 0:
            logger.info(
                ", ".join([
                    "Policy Learning",
                    f"iter={iter_:03d}",
                    f"policy loss={policy_loss_avg.compute():.4f}",
                    f"entropy={entropy_mavg.compute():.4f}",
                    f"logp={logp_mavg.compute():.4f}",
                ])
            )
            writer.add_scalar("policy_learning/loss", policy_loss_avg.compute(), iter_)
            writer.add_scalar("policy_learning/entropy", entropy_mavg.compute(), iter_)
            writer.add_scalar("policy_learning/logp", logp_mavg.compute(), iter_)
            writer.add_scalar("policy_learning/reward", score_avg.compute(), iter_)

        """
        4. evaluate并更新baseline, 找到当前最好的结构
        """
        if iter_ % args.evaluate_controller_freq == 0:
            baseline_arch, best_collect_archs = derive(iter_, controller, nac, 10,
                                                       database, writer, best_collect_archs,
                                                       random_baseline, search_space)
            torch.save(controller.state_dict(), os.path.join(args.output,f"controller-{iter_}.path"))


def generate_architecture_with_pseudo_labels(nac, controller, total, k):
    with torch.no_grad():
        arch_seqs = [controller()[:-2] for _ in range(total)]                           # 用controler生成total个结构
        sample_archs = [seq2arch_fn(seq) for seq in arch_seqs]
        arch0 = [tensorize_fn(arch, device=device) for arch in sample_archs]            # tensor化和pad处理后的结构

        arch0 = batchify(arch0)                                                         # batch处理，返回的list是[邻接矩阵(512,7,7), 算子编码(512,7)]
        if not isinstance(arch0, (list, tuple)):
            arch0 = [arch0]
        arch1 = shuffle(*arch0)                                                         # 对arch0洗一下牌
        # import ipdb; ipdb.set_trace()
        p = nac(arch0, arch1)                                                           # 计算结构0好于打乱的0即结构1的概率, p = torch.Size([512])
        select_p, index = torch.topk(p, k=k)                                            # 选择置信度最高的k个的置信度select_p和其index: select_p = size(256) = index

        arch0 = list_select(arch0, index)
        arch1 = list_select(arch1, index)
        labels = (select_p > 0.5).float()                                               # 置信度select_p大于0.5的其标签为1

    return arch0, arch1, labels


def derive(iter_, controller: NASBenchController, nac: NAC, n_derive: int,
           database, writer, best_collect_archs, random_baseline=False, search_space=None):
    controller.eval()
    with torch.no_grad():
        arch_seqs = [controller()[:-2] for _ in range(n_derive)]                        # 用更新好参数的controller采样n_derive=10个网络结构
        sample_archs = [seq2arch_fn(seq) for seq in arch_seqs]
        arch_tensor = [tensorize_fn(arch, device=device) for arch in sample_archs]

    if random_baseline:
        location = random.choice(list(range(len(arch_tensor))))                         # 从[0-9]采样一个结构的location index
    else:
        outputs = cartesian_traverse(arch_tensor, arch_tensor, nac)                     # 对生成的10个网络，自己和自己做笛卡尔积生成100对结构对，用nac处理得到每个结构对好坏的概率作为输出
        outputs.fill_diagonal_(0)                                                       # outputs = tensor[10,10],这一步将对角线清0，因为自己和自己比不分好坏
        max_p, location = outputs.sum(dim=1).max(dim=0, keepdim=True)                   # 求每一行的和的最大值及其对应的行数，即这一个结构比剩下九个好的概率最大
        max_p = max_p.view([]).item() / n_derive                                        # 除以10求平均
        location = location.view([]).item()

    if database is not None:
        arch = database.fetch_by_spec(sample_archs[location])                           # 本轮最好的采样结果：取出最好的采样的10个里，经过笛卡尔积生成的数据集中对比后，最好的结构
        writer.add_scalar("policy_learning/besttop", arch.rank/database.size*100, iter_)

    history_arch_seqs = arch_seqs + best_collect_archs                                  # 刚训完的controller采样的10个子网和历史最佳采样网络合并为历史网络集合           
    history_arch_tensor = arch_tensor + [tensorize_fn(seq2arch_fn(arch), device=device) for arch in best_collect_archs]
    his_outputs = cartesian_traverse(history_arch_tensor, history_arch_tensor, nac)     # 在历史网络集合中再笛卡尔一下，做成pair数据集，并输出对应nac比较值
    his_outputs.fill_diagonal_(0)                                                       
    his_max_p, his_location = his_outputs.sum(dim=1).max(dim=0, keepdim=True)
    his_max_p = his_max_p.view([]).item() / (n_derive-1)
    his_location = his_location.view([]).item()

    global best_iters
    if his_location < n_derive:                                                         # 本轮采集n_derive个结构，末尾append一个历史最佳，若本轮前n_derive找到了更好的，其结构index一定小于n_derive, 找到最佳的结构数的iter为本iter
        best_iters = iter_

    if database is not None:
        his_best_arch = database.fetch_by_spec(seq2arch_fn(history_arch_seqs[his_location]))    # 取出历史最佳结构：取出全部eps中的最佳的结构
        writer.add_scalar("policy_learning/history_top", his_best_arch.rank/database.size*100, iter_)

    if search_space == "nasbench":
        logger.info(
            ", ".join([
                "DERIVE",
                f"iters={iter_}",
                f"derive {n_derive} archs, the best arch id = {location:02d}, p={max_p*100:.2f}%",
                f"test acc={arch.test_accuracy*100:.2f}%",
                f"rank={arch.rank}/{database.size}({arch.rank/database.size*100:.4f}%)",
                f"history best sampled in {best_iters} iters",
                f"test acc={his_best_arch.test_accuracy*100:.2f}%",
                f"rank={his_best_arch.rank}/{database.size}({his_best_arch.rank/database.size*100:.4f}%)",
            ])
        )
    elif search_space == "darts":
        best_geno = Genotype.from_ordinal_arch(ordinal_normal_arch=arch_seqs[location][0],
                                               ordinal_reduced_arch=arch_seqs[location][1],
                                               primitives=PRIMITIVES)
        his_best_geno = Genotype.from_ordinal_arch(ordinal_normal_arch=history_arch_seqs[his_location][0],
                                                   ordinal_reduced_arch=history_arch_seqs[his_location][1],
                                                   primitives=PRIMITIVES)
        logger.info(
            ", ".join([
                f"DERIVE",
                f"iters={iter_}",
                f"derive {n_derive} archs, the best arch id = {location:02d}, p={max_p*100:.2f}%",
                f"genotype={best_geno}",
                f"history best sampled in {best_iters} iters",
                f"genotype={his_best_geno}",
            ])
        )
    elif search_space == "mobilespace":
        best_arch = arch_seqs[location]
        his_best_arch = history_arch_seqs[his_location]
        logger.info(
            ", ".join([
                f"DERIVE",
                f"iters={iter_}",
                f"derive {n_derive} archs, the best arch id = {location:02d}, p={max_p*100:.2f}%",
                f"best_arch={best_arch}",
                f"history best sampled in {best_iters} iters",
                f"his_best_arch={his_best_arch}",
            ])
        )
    os.makedirs(os.path.join(args.output, "controllers"), exist_ok=True)
    torch.save(controller.state_dict(), os.path.join(args.output, "controllers", f"controller-{iter_}.pth"))

    return [arch_tensor[location]], [history_arch_seqs[his_location]]
