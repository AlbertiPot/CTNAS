"""
target: 读取NAS_Bench_101的json文件
"""
import os
import json

if __name__ == "__main__":
    dataFolder = "/home/gbc/workspace/CTNAS/ctnas/data"
    dataName = "nas_bench.json"
    dataPath = os.path.join(dataFolder, dataName)

    with open(dataPath, 'r') as f:
        dataObject = json.load(f)
    
    print(type(dataObject))
    """
    <class 'list'>
    """
    print(len(dataObject))
    """
    423624
    """
    print(dataObject[0])
    """
    {
        'matrix': [
                [0, 1, 0, 0, 1, 1, 0], 
                [0, 0, 1, 0, 0, 0, 0], 
                [0, 0, 0, 1, 0, 0, 1], 
                [0, 0, 0, 0, 0, 1, 0], 
                [0, 0, 0, 0, 0, 1, 0], 
                [0, 0, 0, 0, 0, 0, 1], 
                [0, 0, 0, 0, 0, 0, 0]], 
        'ops': ['input', 'conv3x3-bn-relu', 'maxpool3x3', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'output'], 
        'hash_': '00005c142e6f48ac74fdcf73e3439874', 
        'n_params': 8555530, 
        'training_time': 1768.7849527994792, 
        'train_accuracy': 1.0, 
        'validation_accuracy': 0.9264155824979147, 
        'test_accuracy': 0.9206063151359558
        }
    """
    
    # for i in range(10):
    #     print(dataObject[i],'\n')



