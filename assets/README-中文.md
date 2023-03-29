# 使用Pytorch训练ResNet实现ImageNet图像分类

Switch Language: [[English](../README.md)]

> Forked from https://github.com/pytorch/examples/tree/main/imagenet

![ImageNet](./imagenet_banner.jpeg)

该代码库主要来源于pytorch官方在Github上发布的example库。建立本库主要是为了熟悉大规模数据集（ImageNet）的处理和加载，以及多卡分布式训练的基本方法。如果需要查看一些更基本的神经网络搭建、模型训练过程的这些代码，请参考[https://github.com/Mr-Philo/Pytorch_LeNet_MNIST](https://github.com/Mr-Philo/Pytorch_LeNet_MNIST)

## ImageNet数据集

下载ImageNet数据集的地址为：[http://www.image-net.org/](http://www.image-net.org/)

下载后，需要解压缩为下列格式的全图片文件夹：

```txt
imagenet/train/
├── n01440764
│   ├── n01440764_10026.JPEG
│   ├── n01440764_10027.JPEG
│   ├── ......
├── ......
imagenet/val/
├── n01440764
│   ├── ILSVRC2012_val_00000293.JPEG
│   ├── ILSVRC2012_val_00002138.JPEG
│   ├── ......
├── ......
```

官方提供了对从ImageNet官网下载下来的压缩包进行解压缩成上面文件的脚本，参考[https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh)

如果所使用的数据集是`tar`类型的压缩包，数据集形式为：

```txt
imagenet
├── train.tar
└── val.tar
```

则可以利用`pip install timm`库对`tar`文件在代码中直接解压load进内存，不需要手动解压，十分方便。这部分代码已经在库里集成完毕。

如果你不想load进真正的ImageNet数据集，仅作代码正确性测试的话，可以在运行文件时加上“使用虚假数据”的选项：`--dummy`，具体用法在下面的部分会介绍

## 启动训练

不考虑分布式计算，假设数据集路径为/data/imagenet，那么最简单的启动脚本为：

```sh
python main.py /data/imagenet
```
或者不使用数据集，而仅用虚假数据做测试：

```sh
python main.py --dummy
```

此时所有的训练设置都会按默认设置来，如模型默认选用resnet18，学习率默认为0.01，每30个epoch衰减至原来的1/10，epoch数量默认为90，batchsize默认为256。这些配置都可以更改，如：

```sh
python main.py -a alexnet --lr 0.01 --epochs 60 --batch-size 128 --output_dir "./my_output" /data/imagenet
```

所有的可选参数项为：

```txt
python main.py
    -a <Arch> 或 --arch <Arch>:  指定模型结构，<Arch>替换为想要训练的模型。可选项有：alexnet | resnet101 | resnet152 | resnet18 |
                        resnet34 | resnet50 | resnext101_32x8d | resnext50_32x4d | vgg11 | vgg 13 | vgg16 | efficientnet | densenet 等等（可参考源码）
    --epochs <N>:  指定运行epochs数量
    -b <N> 或 --batch-size <N>:  指定装载数据集的batch size
    --lr <LR> 或 --learing-rate <LR>:  指定初始学习率
    --momentum <M>:  指定动量
    --wd <W> 或 --weight-decay <W>:  指定权重衰减因子
    --p <N> 或 --print-freq <N>:  log上打印信息的频率，默认为10
    --resume <PATH>:  目前最新checkpoint的路径
    -e 或 --evaluation:  对模型在验证集上做评估
    --pretrained:  使用预训练好的模型
    --seed <SEED>:  将初始种子固定
    --dummy:  使用虚假数据
    --enable_wandb:  使用wandb来记录log，需要`pip install wandb`
    --output_dir:  指定最终模型checkpoint的存放路径
```

还有一部分参数是涉及分布式计算的，统一放到接下来一部分讲

## 分布式计算

> Ref: https://zhuanlan.zhihu.com/p/113694038

这个代码库采用的是`torch.nn.parallel.DistributedDataParallel`框架实现分布式计算，并采用`torch.multiprocessing.spawn`方法启动分布式训练的。另一种更常见的驱动程序的方法是`torch.distributed.launch`，但新版已经被替换成了`torch run`

普通单卡训练的脚本已经在上面提供，这里着重介绍两种分布式计算的方法：单机多卡和多机多卡

### 单机多卡

所谓单机多卡，是指只使用一台机器（node节点数量为1），这台机器上有多张显卡，比较常见的设置为8张卡。利用这台单机多卡进行分布式计算的运行脚本为：

```sh
python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /data/imagenet
```

其中：

`--dist-url`

这个参数是`torch.nn.parallel.DistributedDataParallel`框架在初始化分布式系统时所需要的参数之一。
             
这里有多种初始化方式，一种是采用`env://`开头使用环境变量初始化方式，另一种是共享文件系统初始化方式，这里是采用TCP初始化方式

其中，`127.0.0.1`指主机IP地址，由于是单机训练则用本机地址的相对地址即可（多机时，需要详细填写每一台机器的IP地址）

`FREEPORT`指端口号，这里需要换成一个机器上空闲的端口，避开一些常用端口号如`22`即可，比如可以填为`23456`

`--dist-backend`

指定单机内多卡间的通信方式，采用'nccl'框架是最高效的，当然也可以选择其他的协议如`mpi`,`gloo`等

`--multiprocessing-distributed` 

告诉程序我们采用了分布式训练

`--world-size`

分布式并行中的节点数，即`node`的数量，单机多卡时指定为1

`--rank`

rank在分布式并行中是一个非常重要的概念，指代节点的序号，以帮助程序知道当前的代码运行在哪一块卡上，以确保通信过程中数据流通正确

这里的rank指定为0，指代主机即master machine，由于我们是单机训练，因此这台机器就是主机。

接下来，程序内部会自动根据world-size和rank值来给每一个机器内的每一块卡计算他们的rank值即local-rank, 计算公式为`args.rank = args.rank * ngpus_per_node + gpu`，其中`ngpus_per_node`一般为8，程序内部会调用`ngpus_per_node = torch.cuda.device_count()`自动计算；`gpu`为当前GPU序号，也在程序内部自动计算

### 多机多卡

此时需要明确那一台机器（或者说哪一个node）是主机，并且指导所有机器的IP地址。运行指令为：

Node 0:

```bash
python main.py -a resnet50 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 0 /data/imagenet
```

Node 1:

```bash
python main.py -a resnet50 --dist-url 'tcp://IP_OF_NODE1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 1 /data/imagenet
```

注意，这里指定两个node，则需要依次指定两台机器的IP地址，并将--world-size设置为2，主机的rank设为0，其余机器的rank设为1，2，3，以此类推。