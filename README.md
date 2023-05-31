# cifar100
### 数据集准备
首先下载数据集链接：https://pan.baidu.com/s/1ZR7MKw5j8tPw1E-KEbvD7Q 提取码：k0s5 并存放成如下目录
* cifar100
  * code
  * data
    * cifar-100-python
      * train
      * test
      * meta
### 训练模型 
训练简单卷积网络模型  
```bash
cd code
python train_sim.py
```
训练resnet模型  
```bash
cd code
python train_res.py
```
训练cutmix模型  
```bash
cd code
python train_cutmix.py
```

### 查看训练结果
```bash
cd code/runs
tensorboard --logdir runs 
```
