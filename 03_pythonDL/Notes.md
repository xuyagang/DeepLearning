[TOC]

# Python 深度学习

## 第一部分 深度学习基础

### 第一章   什么是深度学习

#### 1.1人工智能、机器学习、深度学习

![001](D:\project\DL\03_pythonDL\img\001.JPG)

==图灵思考了这样一个问题：通用计算机是否能够学习与创新？他得出的结论是“能”==

- 图灵的这个问题引出了一种新的编程范式。在经典的程序设计（即符号主义人工智能的范 式）中，人们输入的是规则（即程序）和需要根据这些规则进行处理的数据，系统输出的是答案 

  ![002](D:\project\DL\03_pythonDL\img\002.JPG)

- 利用机器学习，人们输入的是数据和从这些数据中预期得到的答案，系统输出的是 规则。这些规则随后可应用于新的数据，并使计算机自主生成答案

- 机器学习系统是训练出来的，而不是明确地用程序编写出来的,__将与某个任务相关的许多 示例输入机器学习系统，它会在这些示例中找到统计结构，从而最终找到规则将任务自动化__

- 。__机器学习__与__数理统计__密切相关，但二者在几个重要方面有所不同
  - 机器学习经常用于处理复杂的 大型数据集（比如包含数百万张图像的数据集，每张图像又包含数万个像素），用经典的统计分 析（比如贝叶斯分析）来处理这种数据集是不切实际的
    - 机器学习（尤其是深度学习） 呈现出相对较少的数学理论
    - 是以工程为导向的。这是一门需要上手实践的 学科，想法更多地是靠实践来证明，而不是靠理论推导

- 从数据中学习表示

  > 机器学习模型将输入数据变换为有意义的输出，这是一个从已知的输入和输出示例中进行 “学习”的过程。因此，机器学习和深度学习的核心问题在于有意义地变换数据，换句话说，在 于学习输入数据的有用表示（representation）——这种表示可以让数据更接近预期输出

  什么是表示？这一概念的核心在于以一种不同的方式 来查看数据（即表征数据或将数据编码）。

  - 彩色图像可以编码为rgb（红绿蓝）或hsv(色相、饱和度、明度)，这是对相同数据的两种表示
    - 对于“选择图像中所有红色像素”这个任务，使用 RGB 格式会更简单
    - 对于“降 低图像饱和度”这个任务，使用 HSV 格式则更简单

  __机器学习模型 都是为输入数据寻找合适的表示——对数据进行变换，使其更适合 手头的任务（比如分类任务）。 __
  - 考虑如下坐标轴中的点

    ![003](D:\project\DL\03_pythonDL\img\003.jpg)

    -  开发一个算法，输入点坐标，判断出点的颜色

      - 输入的是点
      - 预期输出的是点的颜色
      - 衡量算法效果好坏的方式是__正确分类的点所占的百分比__

    - 这里使用坐标变换

      ![004](D:\project\DL\03_pythonDL\img\004.jpg)

      - 利用这种 新的表示，用一条简单的规则就可以描述黑 / 白分类问题：“x>0 的是黑点”或“x<0 的是白点”。 这种新的表示基本上解决了该分类问题。

      > 这个例子中我们认为的定义了坐标变换，但是如果我们尝试系统性的搜索各种可能的坐标变换，并用正确分类的点所占的百分比作为反馈信号，那么我们做的就是机器学习，机器学习中的学习指的是，__寻找更好数据表示的自动搜索过程__
      >
      > 所有机器学习算法都包括自动寻找这样一种变换：这种变换可以根据任务将数据转化为更加 有用的表示。这些操作可能是前面提到的坐标变换，也可能是线性投影（可能会破坏信息）、平移、 非线性操作（比如“选择所有x>0 的点”

      机器学习算法在寻找这些变换时通常没有什么 创造性，而仅仅是遍历一组预先定义好的操作，这组操作叫作假设空间（hypothesis space）。 

  - ==机器学习的技术定义：在预先定义好的可能性空间中，利用反馈信号的指引来寻找 输入数据的有用表示。这个简单的想法可以解决相当多的智能任务，从语音识别到自动驾驶都 能解决。==

- 深度学习之“深度”
  - 深度学习是机器学习的一个分支领域：从数据中学习表示的一种新方法，强调从连续的层中进行学习，这些层对应于越来越有意义的表示

  - “深度学习”中的“深度”指 的并不是利用这种方法所获取的更深层次的理解，而是指一系列连续的表示层。数据模型中 包含多少层，这被称为模型的深度（depth）

  - 这一领域的其他名称包括分层表示学习（layered representations learning）和层级表示学习（hierarchical representations learning）

  - 现代深度学习 通常包含数十个甚至上百个连续的表示层，这些表示层全都是从训练数据中自动学习的。与此 相反，其他机器学习方法的重点往往是仅仅学习一两层的数据表示，因此有时也被称为浅层学 习（shallow learning）

  - 分层表示几乎总是通过叫作神经网络（neural network）的模型来学习 得到的。神经网络的结构是逐层堆叠。

    - 案例：一个多层网络如何对数字图像 进行变换，以便识别图像中所包含的数字

      ![005](D:\project\DL\03_pythonDL\img\005.JPG)

      > 这个网络将数字图像转换成与原始图像差别越来越大的表示，而其中关于 最终结果的信息却越来越丰富。你可以将深度网络看作多级信息蒸馏操作：信息穿过连续的过 滤器，其纯度越来越高

      - 深度学习的技术定义：==学习数据表示的多级方法==。这个想法很简单，但事实证明， 非常简单的机制如果具有足够大的规模，将会产生魔法般的效果。

        ![006](D:\project\DL\03_pythonDL\img\006.JPG)

- 深度学习工作原理

  深度神经网络通过一系列简单的数 据变换（层）来实现这种输入到目标的映射

  - 神经网络中每层对输入数据所做的具体操作保存在该层的权重（weight）中，其本质是一 串数字。==每层实现的变换由其权重来参数化==

  - 权重有时也被称为该层的参数（parameter）,学习的意思是__为神经网络的所有层找到一组权重值，使得该网络能够将每个示例输入与其目标正确地一一对应__

    ![007](D:\project\DL\03_pythonDL\img\007.JPG)

  - 想要控制一件事物，首先需要能够观察它。想要控制神经网络的输出，就需要能够衡量该 输出与预期值之间的距离。这是神经网络__损失函数__（loss function）的任务，该函数也叫__目标函数__（objective function）

  - 损失函数的输入是__网络预测值__与__真实目标值__（即你希望网络输出的 结果），然后计算一个距离值，衡量该网络在这个示例上的效果好坏

    ![008](D:\project\DL\03_pythonDL\img\008.JPG)

  - 深度学习的基本技巧是利用这个距离值作为反馈信号来对权重值进行微调，以降低当前示 例对应的损失值。这种调节由优化器（optimizer）来完成，它实现了所谓的反向 传播（backpropagation）算法，这是深度学习的核心算法

    ![009](D:\project\DL\03_pythonDL\img\009.JPG)

  - 一开始对神经网络的权重随机赋值，因此网络只是实现了一系列随机变换。其输出结果自然也和理想值相去甚远，相应地，损失值也很高。但随着网络处理的示例越来越多，权重值也 在向正确的方向逐步微调，损失值也逐渐降低。这就是训练循环（training loop），将这种循环重 复足够多的次数（通常对数千个示例进行数十次迭代），得到的权重值可以使损失函数最小,==这是 一个简单的机制，一旦具有足够大的规模，将会产生魔法般的效果==

#### 1.2 机器学习简史

深度学习不一定总是解决问题的正确工具：有时没有足够的数据，深度学习不适用；有 时用其他算法可以更好地解决问题。

- 概率建模

  > 是统计学原理在数据分析中的应用,其中最有名的算法之一就是朴素贝叶斯算法

  - 朴素贝叶斯是一类基于应用贝叶斯定理的机器学习分类器，它假设输入数据的特征都是独 立的。这是一个很强的假设，或者说“朴素的”假设,其名称正来源于此。
  - logistic 回归（logistic regression，简称logreg），它有时被认为是 现代机器学习的“hello world”。不要被它的名称所误导——logreg 是一种分类算法，而不是回 归算法

- 核方法

  核方法是一组分类算法，其中最有名的就是支持向量机（SVM， support vector machine）

  > SVM 的目标是通过在属于两个不同类别的两组数据点之间找到良好决策边界（decision boundary)来解决分类问题
  >
  > 决策边界可以看作一条直线或一个平面，将训练数据 划分为两块空间，分别对应于两个类别。对于新数据点的分类，你只需判断它位于决策边界的 哪一侧。

  - SVM 通过两步来寻找决策边界。

    1. 将数据映射到一个新的高维表示，这时决策边界可以用一个超平面来表示（如果数据像下图是二维的，那么超平面就是一条直线）

    2. 尽量让超平面与每个类别最近的数据点之间的距离最大化，从而计算出良好决策边界（分 割超平面），这一步叫作间隔最大化（maximizing the margin）。这样决策边界可以很好 地推广到训练数据集之外的新样本。

    > 其基本思想是：要想在新的表示空间中找到良好的决策超平面，你不需要在新空间中直接计算 点的坐标，只需要在新空间中计算点对之间的距离，而利用核函数（kernel function）可以高效 地完成这种计算。核函数是一个在计算上能够实现的操作，将原始空间中的任意两点映射为这 两点在目标表示空间中的距离，完全避免了对新表示进行直接计算。核函数通常是人为选择的， 而不是从数据中学到的——对于 SVM 来说，只有分割超平面是通过学习得到的。 

    > SVM 很难扩展到大型数据集，并且在图像分类等感知问题上的效果也不好。SVM 是一种比较浅层的方法，因此要想将其应用于感知问题，首先需要手动提取出有用的表示（这 叫作特征工程）

- 决策树、随机森林与梯度提升机

  - 决策树（decision tree）是类似于流程图的结构，可以对输入数据点进行分类或根据给定输 入来预测输出值

  - 随机森林（random forest）算法，它引入了一种健壮且实用的决策树学习方法，即 首先构建许多决策树，然后将它们的输出集成在一起。随机森林适用于各种各样的问题—— 对于任何浅层的机器学习任务来说，它几乎总是第二好的算法

  - 梯度提升机（gradient boosting machine）也是将弱预测模型（通常是决策树）集成的机器学习技术

  - 人们必须竭尽全力让初始输入数据更适合用这些方法处理，也必须 手动为数据设计好的表示层。这叫作特征工程

  - 利用深度学习，你可以一次性学习所有特征，而无须自己手动设计。这极大地简化了机器学习 工作流程，通常将复杂的多阶段流程替换为一个简单的、端到端的深度学习模型

    ==梯度提升机，用于浅层 学习问题；深度学习，用于感知问题。用术语来说，你需要熟悉XGBoost 和 Keras==

### 第二章   神经网络的数学基础

简单的数学概念：张量、张量运算、微分、梯度下降等。 

#### 2.1初识神经网络

- 使用Keras库来学习手写数字分类

  > 将手写数字的灰度图像（28 像素×28 像素）划分到10 个类别 中（0~9）。我们将使用 MNIST 数据集，它是机器学习领域的一个经典数据集，其历史几乎和这 个领域一样长，而且已被人们深入研究。这个数据集包含 60 000 张训练图像和 10 000 张测试图 像，由美国国家标准与技术研究院（National Institute of Standards and Technology，即 MNIST 中 的 NIST）在20 世纪80 年代收集得到。你可以将“解决”MNIST 问题看作深度学习的“Hello World”，正是用它来验证你的算法是否按预期运行。

- 在机器学习中，分类问题中的某个类别叫作类（class）。数据点叫作样本（sample）。某 个样本对应的类叫作标签（label）

  > 过拟合(overfit)是指机器学习模型在新数据上的性能往往比在训练数据上要差

##### [keras 安装](https://www.cnblogs.com/bnuvincent/p/7045324.html)

- 创建tensorflow虚拟环境 C:> conda create -n tensorflow python=3.5，今后所有的东西都需要在该虚拟环境里进行，包括安装各种包和keras
- 激活虚拟环境 C:> activate tensorflow 
- [链接](https://www.tensorflow.org/install/#google-colab-tensorflow) pip install tensorflow
- 安装keras 在tensorflow虚拟环境里面，pip install keras
  - ==在jupyter notebook中使用虚拟环境，需要nb_conda==
    - 在虚拟环境中执行`conda install nb_conda`
    - 之后打开jupyter notebook即可选择虚拟环境

#### 2.2神经网络的数据表示

1. 张量（tensor）是数据容器，包含的几乎总是数值数据
2. 矩阵是二维张量，张量是矩阵向任意维度的推广
3. 张量的维度（dimension），通常叫做轴(axis)

- 标量（0D张量）

  - 仅包含一个数字的张量叫作标量（scalar，也叫标量张量、零维张量、0D 张量）
  - 可用 ndim 属性查看张量轴的个数，轴的个数叫rank
    - 标量张量有0个轴

- 向量（1D张量）

  > 数字组成的数组叫作向量（vector）或一维张量（1D 张量）。一维张量只有一个轴。

  ```
  import numpy as np
  x = np.array([12,3,6,14,23])
  ```

  - 向量有5个元素，被称为5D向量，只有一个轴，沿轴有5个维度
  - 5D张量有5个轴，每个轴有任意维度
  - 维度可以表示沿着某个轴上的元素个数，也可以表示张量中轴的个数，对于张量，更准确的说法是5阶张量（阶表示轴的个数）

- 矩阵（2D张量）

  > 向量组成的数组叫作矩阵（matrix）或二维张量（2D 张量）。矩阵有 2 个轴（通常叫作行和 列）
  >
  > 可以将矩阵直观地理解为数字组成的矩形网格
  - 第一个轴上的元素叫作行（row），第二个轴上的元素叫作列（column）。

- 3D 张量与更高维张量 

  > 将多个矩阵组合成一个新的数组，可以得到一个3D 张量
  >
  > 可以将其直观地理解为数字 组成的立方体

  - 将多个3D 张量组合成一个数组，可以创建一个4D 张量，以此类推
  - 深度学习处理的一般 是 0D 到 4D 的张量，但处理视频数据时可能会遇到 5D 张量

- 关键属性

  - 阶---轴的个数，张量的ndim(3D 张量有3 个轴，矩阵有2 个轴)

  - 形状。这是一个整数元组

    - 向量的形状只包含一个 元素，比如 (5,)，而标量的形状为空，即 ()

  - 数据类型（dtype）

    - 的类型可以是 float32、uint8、float64 等。

      > 在极少数情况下，你可能会遇到字符 （char）张量。注意，Numpy（以及大多数其他库）中不存在字符串张量，因为张量存 储在预先分配的连续内存段中，而字符串的长度是可变的，无法用这种方式存储

- 在numpy中操作张量

  - 张量切片

    ```python
    from keras.datasets import mnist
    (train_images, train_labels),(test_images,test_labels) = mnist.load_data()
    my_slice = train_images[10:100]
    print(my_slice.shape)
    >>>(90, 28, 28)
    # 等价于
    my_slice = train_images[10:100,:,:]
    print(my_slice.shape)
    # 等价于
    myslice  = train_images[10:100,0:28,0:28]
    print(my_slice.shape)
    >>>(90, 28, 28)
    >>>(90, 28, 28)
    ```

  - 也可以使用负数索引，裁剪中心14*14的像素

    ```
    myslice = train_images[:,7:-7,7:-7]
    ```

- 数据批量的概念

  > 通常来说，深度学习中所有数据张量的第一个轴（0 轴，因为索引从0 开始）都是样本轴 （samples axis，有时也叫样本维度）。

  - 深度学习模型不会同时处理整个数据集，而是将数据拆分成小批量:

    - 第一个 `batch = train_images[:128]`
    - 下一个 `batch = train_images[128:256]`
    - 第n个 `batch = train_images[n*128,（n+）*128]`

    > 对于这种批量张量，第一个轴（0轴）叫作批量轴（batch axis）或批量维度（batch dimension）。 

- 现实世界中的张量

  - 向量数据：：2D 张量，形状为 (samples, features)
  - 时间序列数据或序列数据：3D 张量，形状为 (samples, timesteps, features)
  - 图像：4D 张量，形状为 (samples, height, width, channels) 或(samples, channels, height, width)
  - 视频：5D 张量，形状为 (samples, frames, height, width, channels) 或(samples, frames, channels, height, width)

- 向量数据

  - 最常见的数据。每个数据点都被编码为一个向量，因此一个数据批 量就被编码为 2D 张量（即向量组成的数组），其中第一个轴是样本轴，第二个轴是特征轴。

    > 人口统计数据集，其中包括每个人的年龄、邮编和收入。每个人可以表示为包含 3 个值 的向量，而整个数据集包含100 000 个人，因此可以存储在形状为 (100000, 3) 的 2D 张量中

    > 文本文档数据集，我们将每个文档表示为每个单词在其中出现的次数（字典中包含 20 000 个常见单词）。每个文档可以被编码为包含20 000 个值的向量（每个值对应于 字典中每个单词的出现次数），整个数据集包含500 个文档，因此可以存储在形状为 (500, 20000) 的张量中

- 时间序列数据或序列数据 

  > 当时间（或序列顺序）对于数据很重要时，应该将数据存储在带有时间轴的3D 张量中。 
  - 股票价格数据集。每一分钟，我们将股票的当前价格、前一分钟的最高价格和前一分钟 的最低价格保存下来。因此每分钟被编码为一个3D 向量，整个交易日被编码为一个形 状为 (390, 3) 的 2D 张量（一个交易日有 390 分钟），而 250 天的数据则可以保存在一 个形状为 (250, 390, 3) 的 3D 张量中。这里每个样本是一天的股票数据。 
  - 推文数据集。我们将每条推文编码为280 个字符组成的序列，而每个字符又来自于128 个字符组成的字母表。在这种情况下，每个字符可以被编码为大小为 128 的二进制向量 （只有在该字符对应的索引位置取值为1，其他元素都为0）。那么每条推文可以被编码 为一个形状为 (280, 128) 的 2D 张量，而包含100 万条推文的数据集则可以存储在一 个形状为 (1000000, 280, 128) 的张量中

- 图像数据

  > 图像通常具有三个维度：高度、宽度和颜色深度

  > 虽然灰度图像（比如MNIST 数字图像） 只有一个颜色通道，因此可以保存在2D 张量中，但按照惯例，图像张量始终都是3D 张量，灰 度图像的彩色通道只有一维。
  >
  > > 如果图像大小为256×256，那么128 张灰度图像组成的批 量可以保存在一个形状为 (128, 256, 256, 1) 
  > >
  > > 128 张彩色图像组成的批量则可以保存在一个形状为 (128, 256, 256, 3) 的张量中

  - 图像张量的形状有两种约定：通道在后（channels-last）的约定（在 TensorFlow 中使用）和 通道在前（channels-first）的约定（在Theano 中使用）

    > Google 的 TensorFlow 机器学习框架将 颜色深度轴放在最后：(samples, height, width, color_depth)。与此相反，Theano 将图像深度轴放在批量轴之后：(samples, color_depth, height, width)。如果采 用 Theano 约定，前面的两个例子将变成 (128, 1, 256, 256) 和 (128, 3, 256, 256)。 Keras 框架同时支持这两种格式

- 视频数据

  > 5D 张量的少数数据类型之一

  > 视频可以看作一系列帧， 每一帧都是一张彩色图像。由于每一帧都可以保存在一个形状为 (height, width, color_ depth) 的 3D 张量中，因此一系列帧可以保存在一个形状为(frames, height, width, color_depth) 的 4D 张量中，而不同视频组成的批量则可以保存在一个5D 张量中，其形状为 (samples, frames, height, width, color_depth)

#### 2.3 神经网络的“齿轮”：张量运算

> 所有计算机程序最终都可以简化为二进制输入上的一些二进制运算（AND、OR、NOR 等）， 深度神经网络学到的所有变换也都可以简化为数值数据张量上的一些张量运算

>  在 Numpy 中可以直接进行逐元素运算，速度非常快

- 广播
- 不同形状张量运算时，较小的张量会被广播（broadcast）,以匹配较大张量的形状，广播包含一下两步
  1.  向较小的张量添加轴（叫作广播轴），使其 ndim 与较大的张量相同
  2. 将较小的张量沿着新轴重复，使其形状与较大的张量相同

- 张量点积

  > 点积运算，也叫张量积（tensor product，不要与逐元素的乘积弄混）
  >
  > 与逐元素的运算不同，它将输入张量的元素合并在一起。 





















































### 第三章   神经网络入门

### 第四章   机器学习基础

## 第二部分 深度学习实践

### 第五章   深度学习用于计算机视觉

### 第六章   深度学习用于文本和序列

### 第七章   高级深度学习最佳实践

### 第八章   生成式深度学习

### 第九章   总结
