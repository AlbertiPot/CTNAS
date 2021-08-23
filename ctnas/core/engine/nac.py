import random
import functools

import torch
import torch.nn.functional as F

from core.metric import AccuracyMetric, AverageMetric
from core.utils import *


def train(epoch, labeled_loader, pseudo_set, pseudo_ratio, nac, criterion, optimizer, report_freq=10):
    nac.train()
    accuracy = AccuracyMetric()
    loss_avg = AverageMetric()

    for iter_, (*arch, acc, _) in enumerate(labeled_loader, start=1):
    # dataloader提供的数据是(matrix, ops, val_acc, test_acc), 其中*arch保住了matrix，ops，acc指代的val_acc, 这里iter是enumerate提供的
    # arch -> list [(256,7,7), (256,7)] 第一个tensor是256个邻接矩阵，第二个tensor是256个结构的节点算子编码
        *arch0, acc0 = to_device(*arch, acc)
        *arch1, acc1 = shuffle(*arch0, acc0)                                        # 洗一下牌
        targets = (acc0 > acc1).float()
        """
        注意这里的标签构造方法：
        一个batch的结构数据，经过一次随机的洗牌，将洗牌前和洗牌后的acc做一次对比，大于的为1，小于的为0
        这里会不会引入问题，比如对比器见到的数据仅仅是自己batch中的数据
        """
        if pseudo_set is not None and pseudo_ratio != 0:
            batch_size = int(acc0.shape[0] * pseudo_ratio)                          # 扩张batch_size为pseudo_ratio倍
            pseudo_set_size = len(pseudo_set[0][0])
            index = random.sample(list(range(pseudo_set_size)), batch_size)         # 从伪集中选择batch_size大小的index
            un_arch0, un_arch1, pseudo_labels = pseudo_set
            un_arch0 = list_select(un_arch0, index)                                 # 根据index抽取相应的结构和其对应的信息
            un_arch1 = list_select(un_arch1, index)
            pseudo_labels = pseudo_labels[index]
            # import ipdb; ipdb.set_trace()
            arch0 = concat(arch0, un_arch0)
            arch1 = concat(arch1, un_arch1)
            targets = torch.cat([targets, pseudo_labels], dim=0)

        optimizer.zero_grad()

        outputs = nac(arch0, arch1)                                                 # nac对比的两个结构是自己batch中的结构和其打乱之后的结构

        loss = criterion(outputs, targets)

        loss_avg.update(loss)
        accuracy.update(targets, outputs)

        loss.backward()
        optimizer.step()

    logger.info(
        ", ".join([
            "TRAIN Complete",
            f"epoch={epoch:03d}",
            f"accuracy={accuracy.compute()*100:.4f}%",
            f"loss={loss_avg.compute():.4f}",
        ])
    )
    return accuracy.compute(), loss_avg.compute()


def evaluate(epoch, loader, nac):
    nac.eval()

    KTau = AverageMetric()

    for iter_, (*arch, acc, _) in enumerate(loader, start=1):
        *arch, acc = to_device(*arch, acc)

        KTau_ = compute_kendall_tau_AR(nac, arch, acc)

        KTau.update(KTau_)

        logger.info(
            ", ".join([
                "EVAL Complete" if iter_ == len(loader) else "EVAL",
                f"epoch={epoch:03d}",
                f"iter={iter_:03d}",
                f"KTau={KTau_:.4f}({KTau.compute():.4f})",
            ])
        )
    return KTau.compute()
