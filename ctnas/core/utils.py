import os
import sys
import logging
import random

import numpy as np
import torch

from .config import args

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def mean(items):
    return sum(items)/len(items)


def max_with_index(values):
    best_v = values[0]
    best_i = 0
    for i, v in enumerate(values):
        if v > best_v:
            best_v = v
            best_i = i
    return best_v, best_i


def shuffle(*items):
    example, *_ = items
    batch_size, *_ = example.size()
    index = torch.randperm(batch_size, device=example.device)
    
    return [item[index] for item in items]


def to_device(*items):
    return [item.to(device=device) for item in items]


def set_reproducible(seed=0):
    '''
    To ensure the reproducibility, refer to https://pytorch.org/docs/stable/notes/randomness.html.
    Note that completely reproducible results are not guaranteed.
    '''
    random.seed(seed)                           # python seed
    np.random.seed(seed)                        # numpy seed
    torch.manual_seed(seed)                     # torch seed for all devices(both CPU&CUDA)
    torch.backends.cudnn.deterministic = True   # 跨程序运行时选用同一个conv
    torch.backends.cudnn.benchmark = False      # cudnn中每次程序运行中多步调用卷积：跑一次benchmark挑最快的，这里保证不跑benchmarking，全部挑相同的卷积


def get_logger(name: str, output_directory: str, log_name: str, debug: str) -> logging.Logger:
    logger = logging.getLogger(name)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s: %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if output_directory is not None:
        file_handler = logging.FileHandler(os.path.join(output_directory, log_name))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.propagate = False
    return logger


logger = get_logger(name="project", output_directory=args.output, log_name="log.txt", debug=False)


def _sign(number):
    if isinstance(number, (list, tuple)):
        return [_sign(v) for v in number]
    if number >= 0.0:
        return 1
    elif number < 0.0:
        return -1


def compute_kendall_tau(a, b):
    '''
    Kendall Tau is a metric to measure the ordinal association between two measured quantities.
    Refer to https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
    '''
    assert len(a) == len(b), "Sequence a and b should have the same length while computing kendall tau."
    length = len(a)
    count = 0
    total = 0
    for i in range(length-1):
        for j in range(i+1, length):
            count += _sign(a[i] - a[j]) * _sign(b[i] - b[j])
            total += 1
    Ktau = count / total
    return Ktau


def infer(buffer, ranker):
    matrix0 = []
    ops0 = []
    matrix1 = []
    ops1 = []
    for ((matrix0_, ops0_), (matrix1_, ops1_)) in buffer:
        matrix0.append(matrix0_)
        ops0.append(ops0_)

        matrix1.append(matrix1_)
        ops1.append(ops1_)

    matrix0 = torch.stack(matrix0, dim=0)
    ops0 = torch.stack(ops0, dim=0)

    matrix1 = torch.stack(matrix1, dim=0)
    ops1 = torch.stack(ops1, dim=0)

    with torch.no_grad():
        outputs = ranker((matrix0, ops0), (matrix1, ops1))

    return _sign((outputs-0.5).cpu().tolist())


def select(items, index):
    return [items[i] for i in index]


def list_select(items, index):
    return [item[index] for item in items]


def transpose_l(items):
    return list(map(list, zip(*items)))


def index_generate(m, n, up_triangular=False, max_batch_size=1024):
    if up_triangular:
        indexs = []
        for i in range(m-1):
            for j in range(i+1, n):
                indexs.append((i, j))
                if len(indexs) == max_batch_size:
                    yield indexs
                    indexs = []
        if indexs:
            yield indexs
    else:
        indexs = []
        for i in range(m):
            for j in range(n):
                indexs.append((i, j))
                if len(indexs) == max_batch_size:
                    yield indexs
                    indexs = []
        if indexs:
            yield indexs


def batchify(items):
    if isinstance(items[0], (list, tuple)):
        transposed_items = transpose_l(items)
        return [torch.stack(item, dim=0) for item in transposed_items]
    else:
        return torch.stack(items, dim=0) 
    

"""
将arch0和arch1做笛卡尔积，生成len(arch0)*len(arch1)个pair，对其过一下nac输出100个好坏的概率
"""
def cartesian_traverse(arch0, arch1, ranker, up_triangular=False):
    m, n = len(arch0), len(arch1)
    outputs = []
    with torch.no_grad():
        for index in index_generate(m, n, up_triangular):
            i, j = transpose_l(index)
            a = batchify(select(arch0, i))                                      # a是一个bsize为100的邻接矩阵和算子编码的batch，其结构的index是[0*10,1*10,...,9*10]
            b = batchify(select(arch1, j))                                      # b的结构与a相同，但是其index是[(0,1,3,4..,8,9)*10] ，即a和b凑成了笛卡尔积的结果对
            output = ranker(a, b)                                               # 输出100个结构对比后的概率
            outputs.append(output)
    outputs = torch.cat(outputs, dim=0)
    if up_triangular:
        return outputs
    else:
        return outputs.view(m, n)


def compute_kendall_tau_AR(ranker, archs, performances):
    '''
    Kendall Tau is a metric to measure the ordinal association between two measured quantities.
    Refer to https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
    '''
    # assert len(matrix) == len(ops) == len(performances), "Sequence a and b should have the same length while computing kendall tau."
    length = len(performances)
    count = 0
    total = 0

    archs = transpose_l([torch.unbind(item) for item in archs])
    outputs = cartesian_traverse(archs, archs, ranker, up_triangular=True)

    p_combination = _sign((outputs-0.5).cpu().tolist())

    for i in range(length-1):
        for j in range(i+1, length):
            count += p_combination[total] * _sign(performances[i]-performances[j])
            total += 1

    assert len(p_combination) == total
    Ktau = count / total
    return Ktau


def concat(a, b):
    return [torch.cat([item0, item1]) for item0, item1 in zip(a, b)]


def list_concat(a, b):
    if len(a) == 3:
        a0, a1, a2 = a
        b0, b1, b2 = b
        rev0 = concat(a0, b0)
        rev1 = concat(a1, b1)
        rev2 = torch.cat([a2, b2], dim=0)
        return rev0, rev1, rev2
    else:
        a0, a1 = a
        b0, b1 = b
        rev0 = concat(a0, b0)
        rev1 = concat(a1, b1)
        return rev0, rev1
