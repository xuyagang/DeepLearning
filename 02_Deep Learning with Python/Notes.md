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

> __TensorFlow是最著名的用于深度学习生产环境的框架。它有一个非常大非常棒的社区。然而，TensorFlow的使用不那么简单。另一方面，Keras是在TensorFlow基础上构建的高层API，比TF（TensorFlow的缩写）要易用很多。Keras的底层库使用Theano或TensorFlow，这两个库也称为Keras的后端。无论是Theano还是TensorFlow，都是一个“符号式”的库。__
>

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

- 我们通过叠加Dense层来构建网络，Keras层的实例如下所示

  ```python
  keras.layers.Dense(512,activation='relu')
  output = relu(dot(w,input)+b)
  # 这里有三个张量运算
      # 输入张量和张量W之间的的点积运算（dot）
      # 得到的2D张量和向量b之间的加法运算
      # 最后的relu运算，
          # rulu(x)是max(x,0)
  ```

>  在 Numpy 中可以直接进行逐元素运算，速度非常快

- 逐元素运算(是逐元素（element-wise）的运算)

  > 运算独立地应用于张量中的每 个元素，也就是说，这些运算非常适合大规模并行实现

  ```python
  # 想对逐元素运算编写简单的Python 实现，那么 可以用 for 循环。
  # 下列代码是对逐元素 relu 运算的简单实现。
  def naive_relu(x):
      # x是numpy的2d张量
      # 如果不是二维数组就报错
      assert len(x.shape) == 2
      
      x = x.copy()     # 避免覆盖输入张量
      for i in range(x.shape[0]):
          for j in range(x.shape[1]):
              x[i,j] = max(x[i,j],0)
      return x
  # relu(x) 是 max(x, 0)
  
  
  # 对加法采用相同的实现方式
  def naive_add(x,y):
      assert len(x.shape) == 2
      assert x.shape == y.shape
      
      x = x.copy()
      for i in range(x.shape[0]):
          for j in range(x.shape[1]):
              x[i,j] += y[i,j]
      return x
  # 根据同样的方法，你可以实现逐元素的乘法、减法等
  ```

- 广播

- 不同形状张量运算时，较小的张量会被广播（broadcast）,以匹配较大张量的形状，广播包含一下两步
  1.  向较小的张量添加轴（叫作广播轴），使其 ndim 与较大的张量相同
  2. 将较小的张量沿着新轴重复，使其形状与较大的张量相同

- 张量点积

  > 点积运算，也叫张量积（tensor product，不要与逐元素的乘积弄混）
  >
  > 与逐元素的运算不同，它将输入张量的元素合并在一起。 

  - 在 Numpy、Keras、Theano 和 TensorFlow 中，都是用 * 实现逐元素乘积

  - Numpy 和 Keras 中，都是用标准的 dot 运算符来实现点积,TensorFlow 中的 点积使用了不同的语法

    ```python
    import numpy as np 
    z = np.dot(x, y)
    # 数学符号中的点（.）表示点积运算
    z=x.y
    
    # 从数学的角度来看，点积运算做了什么？
    def naive_vector_dot(x,y):
        # x,y是numpy向量
        assert x.shape == 1
        assert y.shape == 1
        assert x.shape[0] == y.shape[0]
        
        z = 0
        for i in range(x.shape[0]):
            z += x[i]*y[i]
        return z
    ```

    - 只有元素个数相同的向量之间才能做点积,==两个向量之间的点积是一个标量==

  - 可以对矩阵x和向量y做点积，返回值是一个向量

    ```python
    # 实现过程
    import numpy as np
    def naive_matrix_vector(x,y):
        assert len(x.shape) == 2
        assert len(y.shape) == 1
        assert x.shape[1] = y.shape[0]
        
        z = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                z[i] += x[i,j] * y[j]
        return z
    ```

    - 如果两个张量中有一个的 ndim 大于1，那么 dot 运算就不再是对称的，也就是说， dot(x, y) 不等于 dot(y, x)

      ```
      a = np.array([[1,2],[1,2]])
      b = np.array([4,5])
      print(np.dot(a,b))
      print(np.dot(b,a))
      >>>[14 14]
      [ 9 18]
      
      a = np.array([1,2,3])
      b = np.array([4,5,6])
      print(np.dot(a,b))
      print(np.dot(b,a))
      >>>32
      32
      ```

    - 点积可以推广到具有任意个轴的张量，最常见的应用可能就是两个==矩阵==之间的点积

      - 对于==两个矩阵 x 和 y==，当且仅==当x.shape[1] == y.shape[0]时==，你才可以对它们做点积---dot(x,y)

        ```python
        # 简单实现
        def naive_matrix_dot(x,y):
            assert len(x.shape) == 2
            assert len(y.shape) == 2
            assert x.shape[1] = y.shape[0]
            
            z = np.zeros(x.shape[0],y.shape[1])
            for i in range(x.shape[0]):
                for j in range(y.shape[1]):
                    row_x = x[i,:]
                    column_y = y[:,j]
                    z[i,j] = naive_vector_dot(row_x,column_y)
        ```

        ![010](D:\project\DL\03_pythonDL\img\010.JPG)

        - x 的行和 y 的列必须大小相同，因 此 x 的宽度一定等于 y 的高度

          > 如果你打算开发新的机器学习算法，可能经常要画这种图。 

        - ==X的列与Y的行数相等，两矩阵的点积就是X的列向量与Y的行向量之间的点积的向量==

        - 可以对更高维的张量做点积，只要其形状匹配遵循与前面2D 张量相同的 原则：

          >(a, b, c, d) . (d,) -> (a, b, c) 
          >
          >(a, b, c, d) . (d, e) -> (a, b, c, e)

- 张量变形（（tensor reshaping）

  > 张量变形是指改变张量的行和列,变形后的张量的元素总个数与初始 张量相同。
  >
  > 一种特殊的张量变形是转置（transposition）,是指将行和列互换， x[i, :] 变为 x[:, i]

- 张量运算的几何解释

  > 对于张量运算所操作的张量，其元素可以被解释为某种几何空间内点的坐标
  >
  > 通常来说，仿射变换、旋转、缩放等基本的几何操作都可以表示为张量运算

- 深度学习的几何解释

  > 神经网络完全由一系列张量运算组成，而这些张量运算都只是输入数据的几何 变换。因此，你可以将神经网络解释为高维空间中非常复杂的几何变换
  - 想象有两张彩纸：一张红色，一张蓝色将其中一张纸放在另一张上。现在将两张纸一起揉成小球。这个皱巴巴的纸球就是你的输入数 据，每张纸对应于分类问题中的一个类别。神经网络（或者任何机器学习模型）要做的就是找 到可以让纸球恢复平整的变换，从而能够再次让两个类别明确可分。

    ![011](D:\project\DL\03_pythonDL\img\011.JPG)

  - 让纸球恢复平整就是机器学习的内容：==为复杂的、高度折叠的数据流形找到简洁的表示==

#### 2.4神经网络的引擎：基于梯度的优化

> - 每个神经层都用下述方法对输入数据进行 变换:
>   output = relu(dot(W, input) + b) 
>
> - W 和 b 都是张量，均为该层的属性。它们被称为该层的权重（weight）或 可训练参数（trainable parameter），分别对应 kernel 和 bias 属性
>
> - 一开始，这些权重矩阵取较小的随机值，这一步叫作随机初始化（random initialization）。
>
>   > W 和 b 都是随机的，relu(dot(W, input) + b) 肯定不会得到任何有用的表示。
>
> - 下一步则是根据反馈信号逐渐调节这些权重。这 个逐渐调节的过程叫作训练
>
> - 上述过程发生在一个训练循环（training loop）内

一般步骤：

1. 抽取训练样本x和对应目标y组成的数据批量

2. 在 x 上运行网络,这一步叫作==前向传播（forward pass）==，得到预测值 y_pred

3. 计算网络在这批数据上的损失，用于衡量 y_pred 和 y 之间的距离。 

4. 更新网络的所有权重，使网络在这批数据上的损失略微下降。 

   > 最终得到的网络在训练数据上的损失非常小，即预测值 y_pred 和预期目标 y 之间的距离 非常小

   - 难点在于第四步，==更新网 络的权重==

     > 一种简单的解决方案是，保持网络中其他权重不变，只考虑某个标量系数，让其尝试不同 的取值。
     >
     > 对于网络中的所有系数都要重复这一过程。 
     >
     > - 这种方法是非常低效的，因为对每个系数（系数很多，通常有上千个，有时甚至多达上 百万个）都需要计算两次前向传播（计算代价很大）

     > 一种更好的方法是利用==网络中所有运算都 是可微（differentiable）的==这一事实，计算损失相对于网络系数的梯度（gradient），然后==向梯度 的反方向改变系数==，从而使损失降低。

##### 2.4.1什么是导数

> 有一个连续的光滑函数 f(x) = y
>
> 假设 x 增大了一个很 小的因子 epsilon_x，这导致 y 也发生了很小的变化，即 :
>
> f(x + epsilon_x) = y + epsilon_y
>
> 在某个点 p 附近，如果 epsilon_x 足够小，就可以将 f 近似为斜率为 a 的线性函数
>
> f(x + epsilon_x) = y + a * epsilon_x

> 斜率 a 被称为 f 在 p 点的导数（derivative）。如果 a 是负的，说明 x 在 p 点的微小变 化将导致 f(x) 减小,
>
> 如果 a 是正的，那么 x 的微小变化将导致 f(x) 增大
>
> - a 的绝对值（导数大小）表示增大或减小的速度快慢

- 对于每个可微函数 f(x)（可微的意思是“可以被求导”)都存在一个导数函数 f'(x)，将 x 的值映射为 f 在该点的局部线性近似的斜率

##### 　2.4.2张量运算的导数：梯度 

> 梯度（gradient）是张量运算的导数。它是导数这一概念向多元函数导数的推广。多元函数 是以张量作为输入的函数

> 对于一个函数 f(x)，你可以通过将 x 向导数的反方向移动一小步来减小 f(x) 的值。同 样，对于张量的函数 f(W)，你也可以通过将 W 向梯度的反方向移动来减小 f(W)，比如 W1 = W0 - step * gradient(f)(W0)，其中 step 是一个很小的比例因子。也就是说，==沿着曲 率的反方向移动，直观上来看在曲线上的位置会更低。==注意，比例因子 step 是必需的，因为 gradient(f)(W0) 只是 W0 附近曲率的近似值，不能离 W0 太远。

##### 2.4.3随机梯度下降

> 给定一个可微函数，理论上可以用解析法找到它的最小值：函数的最小值是导数为0 的点， 因此你只需找到所有导数为 0 的点，然后计算函数在其中哪个点具有最小值

> 将这一方法应用于神经网络，就是用解析法求出最小损失函数对应的所有权重值。可以通 过对方程 gradient(f)(W) = 0 求解 W 来实现这一方法。

1. 抽取训练样本 x 和对应目标 y 组成的数据批量。

2. 在 x 上运行网络，得到预测值 y_pred。

3. 计算网络在这批数据上的损失，用于衡量 y_pred 和 y 之间的距离。

4. 计算损失相对于网络参数的梯度［一次反向传播（backward pass）］。 

5.  将参数沿着梯度的反方向移动一点，比如 W -= step * gradient，从而使这批数据 上的损失减小一点。 

   > 刚刚描述的方法叫作小批量随机梯度下降（mini-batch stochastic gradient descent， 又称为小批量SGD）。术语随机（stochastic）是指每批数据都是随机抽取的（stochastic 是random 在科学上的同义词 a）

- ![012](D:\project\DL\03_pythonDL\img\012.JPG)
  - 直观上来看，为 step 因子选取合适的值是很重要的。如果取值太小，则沿着 曲线的下降需要很多次迭代，而且可能会陷入局部极小点。如果取值太大，则更新权重值之后 可能会出现在曲线上完全随机的位置。 
  - 小批量SGD 算法的一个变体是每次迭代时只==抽取一个样本和目标==，而不是抽取一批 数据。这叫作==真 SGD==（有别于小批量 SGD）。还有另一种极端，每一次迭代都在所有数据上 运行，这叫作==批量 SGD==。这样做的话，每次更新都更加准确，但计算代价也高得多。这两个极 端之间的有效折中则是选择合理的批量大小。 

- SGD 还有多种变体，其区别在于计算下一次权重更新时还要考虑上一次权重更新， 而不是仅仅考虑当前梯度值，比如带动量的SGD、Adagrad、RMSProp 等变体。这些变体被称 为优化方法（optimization method）或优化器（optimizer）。

  - 中动量的概念尤其值得关注，它在 许多变体中都有应用。动量解决了SGD 的两个问题：==收敛速度和局部极小点==

    > ![013](D:\project\DL\03_pythonDL\img\013.JPG)

    > - 在某个参数值附近，有一个局部极小点（local minimum）：在这个点附近，向 左移动和向右移动都会导致损失值增大。如果使用小学习率的SGD 进行优化，那么优化过程可 能会陷入局部极小点，导致无法找到全局最小点。 
    > - 使用动量方法可以避免这样的问题，这一方法的灵感来源于物理学。有一种有用的思维图像， 就是将优化过程想象成一个小球从损失函数曲线上滚下来。如果小球的动量足够大，那么它不会 卡在峡谷里，最终会到达全局最小点。动量方法的实现过程是每一步都移动小球，不仅要考虑当 前的斜率值（当前的加速度），还要考虑当前的速度（来自于之前的加速度）。这在实践中的是指， 更新参数 w 不仅要考虑当前的梯度值，还要考虑上一次的参数更新



- [ ] 不同优化器的了解



##### 2.4.4链式求导：反向传播算法

> 在前面的算法中，我们假设函数是可微的，因此可以明确计算其导数。
>
> 神经网 络函数包含许多连接在一起的张量运算，每个运算都有简单的、已知的导数
>
> - 下面这个 网络 f 包含 3 个张量运算 a、b 和 c，还有 3 个权重矩阵 W1、W2 和 W3
>
>   f(W1, W2, W3) = a(W1, b(W2, c(W3))) 
>
> - 根据微积分的知识，这种函数链可以利用下面这个恒等式进行求导，它称为链式法则（chain rule）： ==(f(g(x)))' = f'(g(x)) * g'(x)==。
>
> - 将链式法则应用于神经网络梯度值的计算，得 到的算法叫作反向传播（backpropagation，有时也叫反式微分，reverse-mode differentiation）



### 第三章   神经网络入门

> - 神经网络的核心组件
> - Keras简介
> - 建立深度学习工作站
> - 使用神经网络解决基本的分类问题和回归问题

- 神经网络最常见的三种 使用场景：==二分类问题、多分类问题和标量回归==
- 神经网络的核心组件:==层、网络、目标函数和优化器==

#### 3.1神经网络剖析

> 训练神经网络主要围绕以下四个方面
>
> - 层，多个层组合成网络（或模型）
> -  输入数据和相应的目标
> - 损失函数，即用于学习的反馈信号
> - 优化器，决定学习过程如何进行

![014](D:\project\DL\03_pythonDL\img\014.JPG)

	> 四者的关系:
	>
	> 多个层连接在一起组成了网络，将输入数据转换为预测值
	>
	> 损失函数将这些预测值与目标进行比较，得到损失值，用于衡量网络预测值与预期结果的匹配程度
	>
	> 优化器使用损失值来更新网络的权重



##### 3.1.1层：深度学习的基础组件

> 神经网络的基本数据结构是==层==。层是一个数据处理模块，将一个 或多个输入张量转换为一个或多个输出张量,
>
> 有些层是无状态的，但大多数的层是有状态的， 即层的权重,
>
> 权重是利用随机梯度下降学到的一个或多个张量

- 简单的向量数据保存在 形状为 (samples, features) 的 ==2D 张量==中，通常用密集连接层［densely connected layer，也 叫全连接层（fully connected layer）或密集层（dense layer），对应于Keras 的 ==Dense== 类］来处 理
- 序列数据保存在形状为 (samples, timesteps, features) 的 ==3D 张量==中，通常用循环 层（recurrent layer，比如Keras 的 ==LSTM 层==）来处理。
- 图像数据保存在==4D 张量==中，通常用二维 卷积层（Keras 的 ==Conv2D==）来处理。 

> 可以将层看作深度学习的乐高积木，Keras 等框架则将这种比喻具体化。

- 在 Keras 中，构 建深度学习模型就是将相互兼容的多个层拼接在一起，以建立有用的数据变换流程。这里层兼 容性（layer compatibility）具体指的是每一层==只接受特定形状的输入张量，并返回特定形状的输 出张量==

```python
from keras import layers 
# 创建了一个层，只接受第一个维度大小为784 的 2D 张量（第0 轴是批量维度，其大 小没有指定，因此可以任意取值）作为输入,该层将返回一个张量，第一个维度大小变成32
layer = layers.Dense(32, input_shape=(784,)) 

# 使用Keras 时，你无须担心 兼容性，因为向模型中添加的层都会自动匹配输入层的形状
from keras import models from keras import layers 
model = models.Sequential() model.add(layers.Dense(32, input_shape=(784,))) model.add(layers.Dense(32))
# 第二层没有输入形状（input_shape）的参数，相反，它可以自动推导出输入形状等于上一层的输出形状
```

- dense层

  ```
  layers.Dense(
      units,
      activation=None,
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      bias_constraint=None,
      **kwargs,
  )
  ```

  - **units:**该层有几个神经元
  - **activation:**该层使用的激活函数
  - **use_bias:**是否添加偏置项
  - **kernel_initializer:**权重初始化方法
  - **bias_initializer:**偏置值初始化方法
  - **kernel_regularizer:**权重规范化函数
  - **bias_regularizer:**偏置值规范化方法
  - **activity_regularizer:**输出的规范化方法
  - **kernel_constraint:**权重变化限制函数
  - **bias_constraint:**偏置值变化限制函数

- 总结：
  - 层对象接受张量为参数，返回一个张量
  - 输入是张量，输出也是张量的一个框架就是一个模型，通过model定义
  - 这样的模型可以被像Keras的sequential一样被训练

##### 3.1.2模型：层构成的网络

> 深度学习模型是层构成的有向无环图。最常见的例子就是层的线性堆叠，将单一输入映射 为单一输出

- 随着深入学习，你会接触到更多类型的网络拓扑结构
  - 双分支（two-branch）网络
  - 多头（multihead）网络
  -  Inception 模块 

- 网络的拓扑结构定义了一个假设空间（hypothesis space）。

  >机器学习 的定义：“在预先定义好的可能性空间中，利用反馈信号的指引来寻找输入数据的有用表示。” 

  选定了网络拓扑结构，意味着将可能性空间（假设空间）限定为一系列特定的==张量运算==，将输 入数据映射为输出数据。然后，你需要为这些张量运算的权重张量找到一组合适的值

##### 3.1.3损失函数和优化器：配置学习过程的关键

- 一旦确定了网络架构，你还需要选择以下两个参数
  - 损失函数（目标函数）——在训练过程中需要==将其最小化==。它能够衡量当前任务是否已 成功完成
  - 优化器——决定如何基于损失函数对网络进行更新。它执行的是==随机梯度下降==（SGD） 的某个变体

- 具有多个输出的神经网络可能具有多个损失函数（每个输出对应一个损失函数），但是，梯 度下降过程必须基于单个标量损失值。因此，对于具有多个损失函数的网络，需要将所有损失 函数取平均，变为一个标量值。

- 一定要明智地选择目标函数，否则你将会遇到意想不到的副作用
  - 二分类问题，你可以使用二元交叉熵（binary crossentropy）损 失函数；
  - 多分类问题，可以用分类交叉熵（categorical crossentropy）损失函数；
  - 序列学习问题，可以用联结主义 时序分类（CTC，connectionist temporal classification）损失函数

#### 3.2 Keras简介

> Keras 是一个Python 深度学习框架，可以方便地定 义和训练几乎所有类型的深度学习模型

- Keras重要特性

  - 相同的代码可以在 CPU 或 GPU 上无缝切换运行
  - 具有用户友好的 API，便于快速开发深度学习模型的原型
  - 内置支持卷积网络（用于计算机视觉）、循环网络（用于序列处理）以及二者的任意 组合。
  - 支持任意网络架构：多输入或多输出模型、层共享、模型共享等。这也就是说，Keras 能够构建任意深度学习模型，无论是生成式对抗网络还是神经图灵机

- batch:批次

  > 深度学习的优化算法，说白了就是梯度下降，每次参数的更新有两种方式：
  >
  > - 遍历全部的数据算一次损失函数，然后算函数对各个参数的梯度，更新梯度
  >   - 这种方法每更新一次参数都要把数据集里的所有样本都看一遍，计算量开销大，计算速度慢，不支持在线学习，这称为Batch gradient descent，批梯度下降
  > - 每看一个数据就算一下损失函数，然后求梯度更新参数，这个称为随机梯度下降，stochastic gradient descent。这个方法速度比较快，但是收敛性能不太好，可能在最优点附近晃来晃去，hit不到最优点
  >
  > 为了克服两种方法的缺点，现在一般采用的是一种折中手段，==mini-batch gradient decent，小批的梯度下降==，这种方法把数据分为若干个批，按批来更新参数，这样，一个批中的一组数据共同决定了本次梯度的方向，下降起来就不容易跑偏，减少了随机性。

- epochs

  > 训练过程中数据将被“轮”多少次

##### 3.2.1Keras、TensorFlow、Theano 和 CNTK 

> Keras 是一个模型级（model-level）的库，为开发深度学习模型提供了高层次的构建模块。 它不处理张量操作、求微分等低层次的运算。相反，它依赖于一个专门的、高度优化的张量库 来完成这些运算，这个张量库就是Keras 的后端引擎（backend engine）。 
>
> Keras 没有选择单个张 量库并将Keras 实现与这个库绑定，而是以模块化的方式处理这个问题（见图3-3）。因此，几 个不同的后端引擎都可以无缝嵌入到 Keras 中。目前，Keras 有三个后端实现：TensorFlow 后端、 Theano 后端和微软认知工具包（CNTK，Microsoft cognitive toolkit）后端。
>
> > TensorFlow、CNTK 和 Theano 是当今深度学习的几个主要平台。Theano 由蒙特利尔大学的 MILA 实验室开发，TensorFlow 由 Google 开发，CNTK 由微软开发
>
> > 用Keras 写的每一段代 码都可以在这三个后端上运行，无须任何修改,这通常是很有用
> >
> > > 例如，对于特定任务，某个后端的速度更快，那么我们就可以无缝切换过去
> > >
> > > - 推荐使用TensorFlow 后端作为大部分深度学习任务的默认后端，因为它 的==应用最广泛，可扩展，而且可用于生产环境==
> > > - 在 CPU 上运行 时，TensorFlow 本身封装了一个低层次的张量运算库，叫作==Eigen==；在GPU 上运行时，TensorFlow 封装了一个高度优化的深度学习运算库，叫作==NVIDIA CUDA==深度神经网络库

##### 3.2.2使用 Keras 开发：概述 

(1) 定义训练数据：输入张量和目标张量。

(2) 定义层组成的网络（或模型），将输入映射到目标。 

(3) 配置学习过程：选择损失函数、优化器和需要监控的指标。 

(4) 调用模型的 fit 方法在训练数据上进行迭代。 

- 定义模型有两种方法：
  - 一种是使用 Sequential 类（仅用于==层的线性堆叠==，这是目前最常 见的网络架构）
  - 另一种是函数式 API（functional API，用于==层组成的有向无环图==，让你可以构 建任意形式的架构）

- 定义好了模型架构，使用 Sequential 模型还是函数式API 就不重要了。接下来的步 骤都是相同的
- 配置学习过程是在编译这一步，你需要指定模型使用的优化器和损失函数，以及训练过程 中想要监控的指标。
- 学习过程就是通过 fit() 方法将输入数据的Numpy 数组（和对应的目标数据）传 入模型

#### 3.3建立深度学习工作站

即使是可以在 CPU 上运行的深度学习应用，使用现代 GPU 通常也可以将速度提高 5 倍或 10 倍。 如果你不想在计算机上安装 GPU，也可以考虑在 AWS EC2 GPU 实例或 Google 云平台上运行深 度学习实验

#### 3.4 电影评论分类：二分类问题

> 二分类问题可能是应用最广泛的机器学习问题,在这个例子中，你将学习根据电影评论的 文字内容将其划分为正面或负面

##### 3.4.1IMDB数据集

- 本节使用IMDB 数据集，它包含来自互联网电影数据库（IMDB）的50 000 条严重两极分 化的评论
- 数据集被分为用于训练的 25 000 条评论与用于测试的 25 000 条评论，训练集和测试 集都包含 50% 的正面评论和 50% 的负面评论

- 与 MNIST 数据集一样，IMDB 数据集也内置于 Keras 库。它已经过预处理：==评论==（单词序列） 已经被转换为==整数序列==，其中每个整数代表字典中的==某个单词==

> imdb.py 的位置：
>
> C:\Users\Dell\Anaconda3\envs\tensorflow\Lib\site-packages\keras\datasets
>
> ```
> # 读取数据时出错
> Object arrays cannot be loaded when allow_pickle=False
> # 打开imdb.py 文件，将np.load(path) 改为np.load(path, allow_pickle=True)
> # 重启kernel
> ```

- 加载数据集

  ```python
  from keras.datasets import imdb
  # num_words的意思是仅保留训练数据中前10000个最常出现的单词，低频词会被舍弃
  # 这样得到的向量数据不会太大，便于处理
  (train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)
  # train_data\test_data是单词索引组成的列表
  # t\rain_labels\test_labels是0或1组成的列表，代表正负面
  # 由于限定为前 10 000 个最常见的单词，单词索引都不会超过 10 000
  max(max(i for i in train_data))
  >>>9995
  # 获取索引和单词的对应字典
  word_index = imdb.get_word_index()
  
  # 把评论转为语句
  # 将评论解码。注意，索引减去了3，
  # 因为0、1、2 是为“padding”（填充）,“ start of sequence”（序 列开始）,“unknown”（未知词）分别保留的索引
  decoded_review = ' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])
  decoded_review
  >>>"? this film was just brilliant casting location scenery story direction everyone's really suited the part they played and you could just imagine being there robert ? is an amazing actor and now the same being director ? father came from the same scottish island as myself so i loved the fact there was a real connection with this film the witty remarks throughout the film were great it was just brilliant so much that i bought the film as soon as it was released for ? and would recommend it to everyone to watch and the fly fishing was amazing really cried at the end it was so sad and you know what they say if you cry at a film it must have been good and this definitely was also ? to the two little boy's that played the ? of norman and paul they were just brilliant children are often left out of the ? list i think because the stars that play them all grown up are such a big profile for the whole film but these children are amazing and should be praised for what they have done don't you think the whole story was so lovely because it was true and was someone's life after all that was shared with us all"
  ```

##### 3.4.2准备数据

> 不能将整数序列直接输入神经网络。你需要将列表转换为张量,方法有以下两种:
>
> - ==填充列表，使其具有相同的长度==，再将列表转换成形状为 (samples, word_indices) 的整数张量，然后网络第一层使用能处理这种整数张量的层( Embedding 层)
> -  对列表进行 ==one-hot 编码[==，将其转换为 0 和 1 组成的向量。举个例子，序列 [3, 5] 将会 被转换为10 000 维向量，只有索引为3 和 5 的元素是1，其余元素都是0。然后网络第 一层可以用 Dense 层，它能够处理浮点数向量数据

```python
# 将整数序列编码为二进制矩阵 
import numpy as np

def vectoriza_sequeces(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_train = vectoriza_sequeces(train_data)
xtest = vectoriza_sequeces(test_data)
# 将标签向量化
print(len(train_labels))
y_train = np.asarray(train_labels,dtype='float32')
y_test = np.asarray(test_labels,dtype='float32')
```

##### 3.4.3构建网络

> 输入数据是向量，而标签是标量（1 和 0），这是你会遇到的最简单的情况
>
> 带有relu 激活的全连接层（Dense）的简单堆叠在这种问题上表现很好,比如：
>
> Dense(16, activation='relu')
>
> -  Dense 层的参数（16）是该层隐藏单元的个数
>   - 一个隐藏单元（hidden unit）是该层 表示空间的一个维度

- 每个relu激活的Dense层都实现了下列张量运算：

  `output = relu(dot(w,inpyt)+b)`

- 16 个隐藏单元对应的权重矩阵 W 的形状为 (input_dimension, 16)，与 W 做点积相当于 将输入数据投影到16 维表示空间中

  > 可以将表 示空间的维度直观地理解为“网络学习内部表示时所拥有的自由度”。隐藏单元越多（即更高维 的表示空间），网络越能够学到更加复杂的表示，但网络的计算代价也变得更大，而且可能会导 致学到不好的模式

- 对于Dense层的堆叠，要确定以下两个关键架构

  - 网络有都少层
  - 每层有多少个隐藏单元

  > 暂时选择下列架构
  >
  > - 两个中间层，每层都有16个隐藏单元
  > - 第三层输出一个标量，预测当前评论的情感

- 中间层使用relu作为激活函数，最后一层使用sigmoid激活以输出一个0~1范围内的概率值（表示样本的目标值等于1的可能性，即为正面的可能性）

  - relu(rectified linear unit 整流线性单元) 函数==将所有负值归零===

  - sigmoid函数则==将任意值’压缩‘到[0,1]区间==内，其输出值可看作是概率值

  - Keras实现

    ![015](D:\project\DL\03_pythonDL\img\015.JPG)

  ```python
  # 总结
  # 快速开始Sequential 模型
  from keras.models import Sequetial
  model = Sequential()
  # 将网络层通过 .add() 堆叠起来，就构成了一个模型
  from keras import Dense,Activation
  model.add(Dense(units=64, input_dim=100))
  model.add(Activation('relu'))
  model.add(Dense(units=10))
  model.add(Activation('softmax'))
  # 完成模型搭建后，需要使用 .compile()方法编译模型
  # 输入损失函数，优化器，指标列表
  model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
  # 完成模型编译后，我们在训练数据上按batch进行一定次数的迭代来训练模型
  model.fit(x_tarin, y_train, epochs=5, batch_size=32)
  ```

- 什么是激活函数？

  >除去relu,Dense 层将只包含两个线性运算——点积 和加法，这样 Dense 层就只能学习输入数据的线性变换（仿射变换），该层的假设空间是从输 入数据到16 位空间所有可能的线性变换集合。这种假设空间非常有限，无法利用多个表示 层的优势
  >
  >为了得到更丰富的假设空间，从而充分利用多层表示的优势，你需要添加非线性或激 活函数。relu 是深度学习中最常用的激活函数

- 最后需要选择损失函数和优化器，==面对二分类问题，网络输出值是一个概率值==（最后一层使用sigmoid 激活函数，仅包含一个单元）,那么最好使用==binary_crossentropy(二元交叉熵)损失==，还可以使用mean_squared_error（均方误差），对于输出概率值的模型，交叉熵（crossentropy）往往是最好 的选择

  > 交叉熵是来自于信息论领域的概念，用于衡量概率分布之间的距离

- 电影评论分类：二分类问题

  - 案例（待总结）

    ```
    
    ```

    

  - 进一步的试验

    - 案例使用了两个隐藏层，可以尝试使用一个或三个隐藏层，然后观察对验证精度和测试精度的影响
    - 尝试使用更多或更少的隐藏单元，比如32、64
    - 尝试使用mse损失函数代替binary_corssentropy
    - 尝试使用tanh激活函数（神经网络早期非常流行）代替relu

  - 小结（学习要点）

    - 通常需要对原始数据进行大量预处理，以便将其转换为张量输入到神经网络中。单词序 列可以编码为二进制向量，但也有其他编码方式
    - 带有 relu 激活的 Dense 层堆叠，可以解决很多种问题（包括情感分类），你可能会经 常用到这种模型
    - 对于二分类问题（两个输出类别），网络的最后一层应该是只有一个单元并使用 sigmoid 激活的 Dense 层，网络输出应该是 0~1 范围内的标量，表示概率值
    - 对于二分类问题的 sigmoid 标量输出，你应该使用 binary_crossentropy 损失函数
    - 无论你的问题是什么，rmsprop 优化器通常都是足够好的选择。这一点你无须担心
    - 随着神经网络在训练数据上的表现越来越好，模型最终会过拟合，并在前所未见的数据 上得到越来越差的结果。一定要一直监控模型在训练集之外的数据上的性能


#### 新闻分类：多分类问题

上一节中，我们介绍了如何用密集连接的神经网络将向量输入划分为两个互斥的类别，

本节你会构建一个网络，将路透社新闻划分为46 个互斥的主题。因为有多个类别，所以 这是多分类（multiclass classification）问题

因为每个数据点只能划分到一个类别， 所以更具体地说，这是单标签、多分类（single-label, multiclass classification）问题

如果每个数据点可以划分到多个类别（主题），那它就是一个多标签、多分类（multilabel, multiclass classification）问题

- 路透社数据集

  路透社数据集，它包含许多短新闻及其对应的主题，由路透社在1986 年发布。它 是一个简单的、广泛使用的文本分类数据集。它包括46 个不同的主题：某些主题的样本更多， 但训练集中每个主题都有至少 10 个样本

  与 IMDB 和 MNIST 类似，路透社数据集也内置为 Keras 的一部分



compile函数学习补充：

```
compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)

optimizer: 字符串（优化器名）或者优化器实例
loss: 字符串（目标函数名）或目标函数
metrics: 在训练和测试期间的模型评估标准
	通常你会使用 metrics = ['accuracy']
	为多输出模型的不同输出指定不同的评估标准,metrics = {'output_a'：'accuracy'}
loss_weights: 可选的指定标量系数（Python 浮点数）的列表或字典， 用以衡量损失函数对不同的模型输出的贡献
sample_weight_mode: 如果你需要执行按时间步采样权重（2D 权重），请将其设置为 temporal
weighted_metrics: 在训练和测试期间，由 sample_weight 或 class_weight 评估和加权的度量标准列表
```



```python
from keras.datasets import reuters 

# num_words=10000 将数据限定为前 10 000 个最常出现的单词
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(     num_words=10000) 

# 每个样本都是一个整数列表（表示单词索引）
train_data[10] 
>>> [1, 245, 273, 207, 156, 53, 74, 160, 26, 14, 46, 296, 26, 39, 74, 2979, 3554, 14, 46, 4689, 4329, 86, 61, 3499, 4795, 14, 61, 451, 4329, 17, 12]

# 解码索引
word_index = reuters.get_word_index()
reverse_word_index =  dict([(value, key) for (key, value) in word_index.items()]) 
# 索引减去了3，因为0、1、2 是为“padding”（填充）、“ start of sequence”（序列开始）、“unknown”（未知词）分别保留的索引
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in     train_data[0]])

# 样本对应的标签是一个 0~45 范围内的整数，即话题索引编号
train_labels[10] 
>>>3
```

- 准备数据

  将数据向量化：

  ```python
  #  编码数据 
  import numpy as np 
  
  def vectorize_sequences(sequences, dimension=10000):
      results = np.zeros((len(sequences),dimension))
      for i,seq in enumerate(sequences):
          results[i,seq] = 1
      return results
  
  x_train = vectorize_sequences(train_data) 
  x_test = vectorize_sequences(test_data) 
  ```

  将标签向量化有两种方法：

  - 你可以将标签列表转换为整数张量

  - 或者使用one-hot 编码 

    > one-hot 编码是分类数据广泛使用的一种格式，也叫分类编码（categorical encoding）

    ```python
    def to_one_hot(labels,dimension=46):
        results = np.zeros((len(labels),dimension))
        for i,label in enumerete(labels):
            results[i,label] = 1
        return results
    
    # 将训练标签向量化
    one_hot_train_labels = to_one_hot(train_labels) 
    # 将测试标签向量化
    one_hot_test_labels = to_one_hot(test_labels) 
    ```

  Keras 内置方法可以实现这个操作

  ```
  from keras.utils.np_utils import to_categorical
  one_hot_train_labels = to_categorical(train_labels)
  one_hot_test_labels = to_categorical(test_labels)
  ```

- 构建网络

  对文本片段进行分类，输出类别的数量从2 个变为46 个。输出空间的维 度要大得多

  对于Dense层的堆叠，每层只能访问上层输出的信息，如果某层丢失了于分类相关的信息，后面的层无法找回，也就是说每层都可能成为信息瓶颈，处于这个原因，下面使用更大的层，包含64个单元

  ```python
  from keras import models
  from keras import layers
  
  model = models.Sequential()
  model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
  model.add(layers.Dense(64,activation='relu'))
  model.add(layers.Dense(46,activation='softmax'))
  '''
   网络的最后一层是大小为46 的 Dense 层。这意味着，对于每个输入样本，网络都会输 出一个 46 维向量。这个向量的每个元素（即每个维度）代表不同的输出类别,最后一层使用了 softmax 激活,网络将输出在46 个不同输出类别上的概率分布——对于每一个输入样本，网络都会输出一个 46 维向量， 其中 output[i] 是样本属于第 i 个类别的概率。46 个概率的总和为 1。 
   '''
  
  '''对于这个例子，最好的损失函数是 categorical_crossentropy（分类交叉熵）。它用于 衡量两个概率分布之间的距离，这里两个概率分布分别是网络输出的概率分布和标签的真实分 布。通过将这两个分布的距离最小化，训练网络可使输出结果尽可能接近真实标签'''
  ```

- 编译模型

  ```python
  model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  ```

- 验证

  我们在训练数据中留出 1000 个样本作为验证集

  ```python
  # 留出训练验证集
  x_val = x_train(1000:)
  partial_x_train = x_train(:1000)
  # 留出标签验证集
  y_val = one_hot_train_labels[:1000]
  partial_y_train = one_hot_train_labels
  ```

- 训练模型

  ```python
  history = model.fit(partial_x_train,
                     partial_y_train,
                     epochs=20,
                     batch_size=512,
                     validation_data=(x_val,y_val))
  ```

- 绘制训练损失和验证损失、训练精度和验证精度的图像

  ```python
  # 训练损失和验证损失
  import matplotlib.pyplot as plt 
   
  loss = history.history['loss'] val_loss = history.history['val_loss'] 
   
  epochs = range(1, len(loss) + 1) 
   
  plt.plot(epochs, loss, 'bo', label='Training loss') plt.plot(epochs, val_loss, 'b', label='Validation loss') plt.title('Training and validation loss') plt.xlabel('Epochs') plt.ylabel('Loss') plt.legend() 
   
  plt.show()
  
  # 训练精度和验证精度
  acc = history.history['acc'] val_acc = history.history['val_acc'] 
   
  plt.plot(epochs, acc, 'bo', label='Training acc') plt.plot(epochs, val_acc, 'b', label='Validation acc') plt.title('Training and validation accuracy') plt.xlabel('Epochs') plt.ylabel('Accuracy') plt.legend() 
   
  plt.show() 
  ```

  

#### 3.6 预测房价：回归问题

前面两个例子都是分类问题，其目标是预测输入数据点所对应的单一离散的标签。另一种 常见的机器学习问题是回归问题，它预测一个连续值而不是离散的标签

不要将回归问题与 logistic 回归算法混为一谈，logistic 回归不是回归算法， 而是分类算法

- 数据集

  本节将要预测20 世纪70 年代中期波士顿郊区房屋价格的中位数，已知当时郊区的一些数 据点，比如犯罪率、当地房产税率等。本节用到的数据集与前面两个例子有一个有趣的区别。 它包含的数据点相对较少，只有506 个，分为404 个训练样本和102 个测试样本。输入数据的 每个特征（比如犯罪率）都有不同的取值范围。例如，有些特性是比例，取值范围为0~1；有 的取值范围为 1~12；还有的取值范围为 0~100

  ```python
  from keras.datasets import boston_housing
  
  # 有404 个训练样本和 102 个测试样本，每个样本都有13 个数值特征
  (train_data,train_targets),(test_data,test_targets) = boston_housing.load_data()
  print(train_data.shape,test_data.shape)
  >404, 13) (102, 13)
  
  The data comprises 13 features. The 13 features in the input data are as 
  follow:
  1. Per capita crime rate.
  2. Proportion of residential land zoned for lots over 25,000 square feet.
  3. Proportion of non-retail business acres per town.
  4. Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
  5. Nitric oxides concentration (parts per 10 million).
  6. Average number of rooms per dwelling.
  7. Proportion of owner-occupied units built prior to 1940.
  8. Weighted distances to five Boston employment centres.
  9. Index of accessibility to radial highways.
  10. Full-value property-tax rate per $10,000.
  11. Pupil-teacher ratio by town.
  12. 1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
  13. % lower status of the population.
  
  The targets are the median values of owner-occupied homes, in thousands of dollars:
  ```

- 准备数据

  - 取值范围差异很大的数据输入到神经网络中，这是有问题的。网络可能会自动适应这种 取值范围不同的数据，但学习肯定变得更加困难

  - 对于这种数据，普遍采用的最佳实践是对每 个特征做==标准化==，即对于输入数据的每个特征（输入数据矩阵中的列），减去特征平均值，再除 以标准差，这样得到的特征平均值为 0，标准差为 1。

    ```python
    import numpy as np
    
    mean = train_data.mean(axis=0)
    print(train_data.shape)
    print(mean.shape)
    
    train_data -= mean
    std = train_data.std(axis=0)
    train_data/=std
    
    test_data -= mean
    test_data /= std
    ```

- 构建网络

  - 由于样本数量很少，我们将使用一个非常小的网络，其中包含两个==隐藏层==，每层有64 个单 元。一般来说，训练数据越少，过拟合会越严重，而较小的网络可以降低过拟合。

    ```python
    from keras import models
    from keras import layers
    
    def build_model():
        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu',
                               input_shape=(train_data.shape[1],)))
        model.add(layers.Dense(64,activation='relu'))
        model.add(layers.Dense(1))
        # metrics: 在训练和测试期间的模型评估标准
        # mae:mean absolute error 差值绝对值求和平均
        # mse:mean square erros 差值平方求和平均
        model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
        return model
    ```

    网络的最后一层只有一个单元，没有激活，是一个线性层。这是标量回归（标量回归是预 测单一连续值的回归）的典型设置。添加激活函数将会限制输出范围。例如，如果向最后一层 添加 sigmoid 激活函数，网络只能学会预测0~1 范围内的值。这里最后一层是纯线性的，所以 网络可以学会预测任意范围内的值

    编译网络用的是 ==mse 损失函数==，即==均方误差==（MSE，mean squared error），预测值与 目标值之差的平方。这是回归问题常用的损失函数

    在训练过程中还==监控一个新指标==：==平均绝对误差==（MAE，mean absolute error）。它是预测值 与目标值之差的绝对值。比如，如果这个问题的MAE 等于0.5，就表示你预测的房价与实际价 格平均相差 500 美元

- K折验证

  ```python
  # k折验证
  import numpy as np
  
  # 组数
  k = 4
  # 求出每组的个数
  num_val_samples = len(train_data)//k
  num_epochs = 100
  all_scores = []
  
  for i in range(k):
      print('processing fold #',i)
      # 第k个分区的数据
      val_data = train_data[i*num_val_samples: (i+1)*num_val_samples]
      val_targets = train_targets[i*num_val_samples:(i+1)*num_val_samples]
      
      # 利用concatenate合并两个分开的数组，作为训练数据
      partial_train_data = np.concatenate(
          [train_data[:i*num_val_samples],
           train_data[(i+1)*num_val_samples:]],
          axis=0)
      
      partial_train_targets = np.concatenate(
          [train_targets[:i * num_val_samples],
           train_targets[(i + 1) * num_val_samples:]],
          axis=0) 
  
      model = build_model()
      model.fit(partial_train_data,
                partial_train_targets,
                epochs=num_epochs,
                batch_size=1,
                # 0, 1 或 2。日志显示模式。 0 = 安静模式, 1 = 进度条, 2 = 每轮一行
                verbose=0)
  
      # evaluate:在测试模式下返回模型的误差值和评估标准值
      val_mse,val_mae = model.evaluate(val_data,val_targets,verbose=0)
      all_scores.append(val_mae)
  ```

  每次运行模型得到的验证分数有很大差异，从2.6 到 3.2 不等。平均分数（3.0）是比单一 分数更可靠的指标——这就是K 折交叉验证的关键

  让训练时间更长一点，达到500 个轮次。为了记录模型在每轮的表现，我们需要修改 训练循环，以保存每轮的验证分数记录

- 保持每折的结果

  ```python
  from keras import backend as K
  
  # Some memory clean-up
  K.clear_session()
  
  num_epochs = 500
  k = 4
  num_val_samples = len(train_data)//k
  all_mae_histories = []
  
  for i in range(k):
      print('processing fold # ',i)
      val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]
      val_targets = train_targets[i*num_val_samples:(i+1)*num_val_samples]
      
      partial_train_data = np.concatenate(
          (train_data[:i*num_val_samples],train_data[(i+1)*num_val_samples:]),
          axis=0)
      partial_train_targets = np.concatenate(
          (train_targets[:i*num_val_samples],train_targets[(i+1)*num_val_samples:]),
          axis=0)
      
      model = build_model()
      history = model.fit(partial_train_data,
                          partial_train_targets,
                          # 用来评估损失，以及在每轮结束时的任何模型度量指标
                          validation_data=(val_data,val_targets),
                          batch_size=1,
                          epochs=num_epochs,
                          verbose=0)
      mae_history = history.history['mean_absolute_error']
      all_mae_histories.append(mae_history)
  ```

- 绘制验证分数

  ```
  import matplotlib.pyplot as plt
  
  plt.plot(range(1,len(average_mae_history)+1),average_mae_history)
  plt.xlabel('Epochs')
  plt.ylabel('Validation MAE')
  plt.show()
  ```

  ![016](D:\project\DL\02_Deep Learning with Python\img\016.jpg)



因为纵轴的范围较大，且数据方差相对较大，所以难以看清这张图的规律。我们来重新绘 制一张图

删除前 10 个数据点，因为它们的取值范围与曲线上的其他点不同

将每个数据点替换为前面数据点的指数移动平均值，以得到光滑的曲线

```python
def smooth_curve(points,factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history  = smooth_curve(average_mae_history[10:])
len(smooth_mae_history)
```

```python
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.plot(range(1,len(average_mae_history)-9),average_mae_history[10:])
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
```

![017](D:\project\DL\02_Deep Learning with Python\img\017.jpg)

完成模型调参之后（除了轮数，还可以调节隐藏层大小），你可以使用最佳参数在所有训练 数据上训练最终的生产模型，然后观察模型在测试集上的性能。

```python
# 一个全新的模型
model = build_model()
model.fit(train_data,train_targets,epochs=80,batch_size=16,verbose=0)
test_mse_score,test_mae_score = model.evaluate(test_data,test_targets)
```

- 要点
  - 回归问题使用的损失函数与分类问题不同。回归常用的损失函数是均方误差（MSE）
  - 回归问题使用的评估指标也与分类问题不同。显而易见，精度的概念不适用于回 归问题。常见的回归指标是平均绝对误差（MAE）
  - 如果输入数据的特征具有不同的取值范围，应该先进行预处理，对每个特征单独进行 缩放。
  - 如果可用的数据很少，使用 K 折验证可以可靠地评估模型。 
  - 如果可用的训练数据很少，最好使用隐藏层较少（通常只有一到两个）的小型网络，以 避免严重的过拟合。

- 总结
  - 现在你可以处理关于向量数据最常见的机器学习任务了：二分类问题、多分类问题和标 量回归问题。前面三节的“小结”总结了你从这些任务中学到的要点
  - 在将原始数据输入神经网络之前，通常需要对其进行预处理。 
  - 如果数据特征具有不同的取值范围，那么需要进行预处理，将每个特征单独缩放
  - 随着训练的进行，神经网络最终会过拟合，并在前所未见的数据上得到更差的结果
  - 如果训练数据不是很多，应该使用只有一两个隐藏层的小型网络，以避免严重的过拟合
  - 如果数据被分为多个类别，那么中间层过小可能会导致信息瓶颈
  - 如果要处理的数据很少，K 折验证有助于可靠地评估模型。



### 第四章   机器学习基础

本章内容：

- 除去分类和回归之外的机器学习形式
- 评估模型的规范流程
- 为深度学习准备数据
- 特征工程
- 解决过拟合
- 处理机器学习问题的通用工作流程

已经知道如何用神经网络解决分类问题和回归问题，而且 也看到了机器学习的核心难题：过拟合。

我们将把所有这些概念——模型评估、数据预处理、特征工程、解决过 拟合——整合为详细的==七步工作流程==，用来解决任何机器学习任务。 

#### 4.1 机器学习的四个分支

已经熟悉了三种类型的机器学习问题：二分类问题、多分类问题和标 量回归问题。这三者都是==监督学习（supervised learning）==的例子，其目标是学习训练输入与训 练目标之间的关系

##### 4.1.1监督学习

监督学习是目前最常见的机器学习类型。__给定一组样本（通常由人工标注），它可以学会将 输入数据映射到已知目标［也叫标注（annotation）］__。

广受关注的深度学习应用几乎都属于监督学习，比如光学字符识别、语音识别、 图像分类和语言翻译

监督学习主要包括分类和回归，但还有更多的奇特变体：

- ==序列生成==（sequence generation）。给定一张图像，预测描述图像的文字。序列生成有时 可以被重新表示为一系列分类问题，比如反复预测序列中的单词或标记
- ==语法树预测==（syntax tree prediction）。给定一个句子，预测其分解生成的语法树
- ==目标检测==（object detection）。给定一张图像，在图中特定目标的周围画一个边界框。这 个问题也可以表示为分类问题（给定多个候选边界框，对每个框内的目标进行分类）或 分类与回归联合问题（用向量回归来预测边界框的坐标）
- ==图像分割==（image segmentation）。给定一张图像，在特定物体上画一个像素级的掩模（mask）。

##### 4.1.2无监督学习

无监督学习是指__在没有目标的情况下寻找输入数据的有趣变换__，其目的在于==数据可视化、 数据压缩、数据去噪或更好地理解数据中的相关性==。无监督学习是数据分析的必备技能，在解 决监督学习问题之前，为了更好地了解数据集，它通常是一个必要步骤。__降维（dimensionality reduction）和聚类（clustering）__都是众所周知的无监督学习方法

##### 4.1.3

自监督学习是监督学习的一个特例，它与众不同，值得单独归为一类。==自监督学习是没有 人工标注的标签的监督学习==，你可以将它看作没有人类参与的监督学习。标签仍然存在（因为 总要有什么东西来监督学习过程），但它们是从输入数据中生成的，通常是使用__启发式算法__生 成的。

自编码器（autoencoder）是有名的自监督学习的例子，其生成的目标就是未经 修改的输入。同样，给定视频中过去的帧来预测下一帧，或者给定文本中前面的词来预测下一个词， 都是自监督学习的例子［这两个例子也属于==时序监督学习（temporally supervised learning）==， 即用 未来的输入数据作为监督］。注意，监督学习、自监督学习和无监督学习之间的区别有时很模糊， 这三个类别更像是没有明确界限的连续体。自监督学习可以被重新解释为监督学习或无监督学 习，这取决于你关注的是学习机制还是应用场景

##### 4.1.4强化学习

强化学习中，智能体（agent）接收有关其环境的信息，并学会选择使某种奖励最大化的行动。 例如，神经网络会“观察”视频游戏的屏幕并输出游戏操作，目的是尽可能得高分，这种神经 网络可以通过强化学习来训练

强化学习主要集中在研究领域，除游戏外还没有取得实践上的重大成功。但是，我 们期待强化学习未来能够实现越来越多的实际应用：自动驾驶汽车、机器人、资源管理、教育等。 



##### 分类回归术语表：

___

- 样本（sample）或输入（input）：进入模型的数据点
- 预测（prediction）或输出（output）：从模型出来的结果
- 目标（target）：真实值。对于外部数据源，理想情况下，模型应该能够预测出目标
- 预测误差（prediction error）或损失值（loss value）：模型预测与目标之间的距离
- 类别（class）：分类问题中供选择的一组标签。例如，对猫狗图像进行分类时，“狗” 和“猫”就是两个类别。
- 标签（label）：分类问题中类别标注的具体例子。比如，如果1234 号图像被标注为 包含类别“狗”，那么“狗”就是 1234 号图像的标签
- 真值（ground-truth）或标注（annotation）：数据集的所有目标，通常由人工收集
- 二分类（binary classification）：一种分类任务，每个输入样本都应被划分到两个互 斥的类别中
- 多分类（multiclass classification）：一种分类任务，每个输入样本都应被划分到两个 以上的类别中，比如手写数字分类
- 多标签分类（multilabel classification）：一种分类任务，每个输入样本都可以分配多 个标签。举个例子，如果一幅图像里可能既有猫又有狗，那么应该同时标注“猫” 标签和“狗”标签。每幅图像的标签个数通常是可变的
- 标量回归（scalar regression）：目标是连续标量值的任务。预测房价就是一个很好的 例子，不同的目标价格形成一个连续的空间
- 向量回归（vector regression）：目标是一组连续值（比如一个连续向量）的任务。如 果对多个值（比如图像边界框的坐标）进行回归，那就是向量回归
- 小批量（mini-batch）或批量（batch）：模型同时处理的一小部分样本（样本数通常 为 8~128）。样本数通常取2 的幂，这样便于GPU 上的内存分配。训练时，小批量 用来为模型权重计算一次梯度下降更新

















































## 第二部分 深度学习实践

### 第五章   深度学习用于计算机视觉

### 第六章   深度学习用于文本和序列

### 第七章   高级深度学习最佳实践

### 第八章   生成式深度学习

### 第九章   总结

