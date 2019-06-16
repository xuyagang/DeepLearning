## 第三章   多层全连接神经网络

### 3.1 PyTorch基础

- Tensor 张量

  - 表示一个多维矩阵，和nympy是对应的，tensor可以和ndarray相互转换，pytorch可以在gpu上运算，numpy的ndarray只能在cpu运行
    - 零维是一个点
    - 一维是一个向量
    - 二维就是一般的矩阵
    - 多维矩阵

- tensor类型

  - torch.FloatTensor     32位浮点型（==默认类型==）

  - torch.DoubleTensor     64位浮点型

  - torch.ShortTensor     16位整型

  - torch.IntTensor     32位整型

  - torch.LongTensor     64位整型

    ```python
    #  torch.Tensor 默认的是 torch.FloatTensor 数据类型
    a = torch.Tensor([[2,3],[4,8],[7,9]])
    print(f'a is:\n{a}')
    >>>tensor([[2., 3.],
            [4., 8.],
            [7., 9.]])
    print(a.size())
    >>>torch.Size([3, 2])
    # 自定义数据类型(64位整数)
    b = torch.LongTensor([[2,3],[4,8],[7,9]])
    ```
    - 全是零的空Tensor和正态分布作为随机起始值

      ```python
      c = torch.zeros((3,2))
      d = torch.randn((3,2))
      ```

    - 索引和改变值的大小

      ```
      a[0,1] = 100
      ```

    - Tensor和numpu.ndarray转换

      ```python
      b = torch.LongTensor([[2,3],[4,8],[7,9]])
      # tensor 转 ndarray
      numpy_b = b.numpy()
      # ndarray 转 numpy
      e = np.array([[2,3],[4,5]])
      torch_e = torch.from_numpy(e)
      ```

    - 如果 你的电脑支持GPU，还能把tensor放到GPU上

      ```python
      # 只需要 a.cuda() 就能将tensor a 放到GPU上了
      if torch.cuda.is_available():
          a_cuda = a.cuda()
          print(a_cuda)
      ```

- Variable 变量

  > 1. variable变量，在numpy里就没有了，是神经网络计算图里特有的一个概念，__就是variable提供了自动求导的功能__
  > 2. variable和tensor没有本质的区别，不过variable会被放入一个计算图中，然后进行前向传播，反向传播、自动求导

  - Variable是在torch.autograd.Variable 中

    > 将tensor变为Variable，只需要 Variable（a）

    - 属性

      1. data     读取Variable中的tensor值
      2. grad     是Variable的反向传播梯度
      3. grad_fn     表示得到这个Variable的操作，比如通过加减还是乘除来得到

      ```python
      # 标量求导
      import torch.autograd.variable as va
      # 创建Variable
      x = va(torch.Tensor([1]),requires_grad = True)
      w = va(torch.Tensor([2]),requires_grad = True)
      b = va(torch.Tensor([3]),requires_grad = True)
      y = w * x + b
      # 自动求导
      y.backward()
      #等价于
      y.backward(torch.FloatTensor([1]))
      ```

      - 构建variable时，参数requires_grad=True表示是否对变量求梯度，默认False

      ```python
      # 矩阵求导
      # 构建tensor
      x = torch.randn(3)
      # 构建variable
      x = va(x,requires_grad=True)
      y = x ** 2
      print(y)
      >>>tensor([1.4519, 0.2551, 0.4936], grad_fn=<PowBackward0>)
      # y是一个向量，这时求导需要传入参数声明，比如：
      y.backward(torch.FloatTensor([1,0.1,0.01]))
      #  y.backward(torch.FloatTensor [1，1， 1 ]) 这样得到的 结果就是它们每个分量的梯度
      #  y.backward(torch.FloatTensor( [1， 0.1 ， 0. 01] )) ，这样得到的梯度就是它们原本的梯度分别乘上 1 ， 0.1 和 0.01
      
      # 求梯度
      y.backward(torch.FloatTensor([1,1,1]))
      ```

- Dataset(数据集)

  > 处理任何机器学习问题之前都要做数据读取，并预处理，PyTorch提供了很多数据的读取和预处理工具
  - torch.utils.data.Dataset是代表这一数据的抽象类，你可以定义你的数据类继承和重写这个类，只需要定义\_\_len\_\_ 和\_\_getitem\_\_

  - 通过torch.utils.data.DataLoader来定义迭代器

    `dataiter = DataLoader(mydataset,batch_size=32,shuffle=True,collate_fn=default_collate)`

    - collate_fn表示如何取样本

  - torchvision包中有一个关于计算机视觉的数据读取类ImageFolder,要求图片是如下形式

    ```
    root/dog/xxx.png
    root/dog/xxy.png
    root/dog/xxz.png
    ```

    调用：

    `dset = ImageFolder(foot='root_path',transform=None,loader=default_loader)`

    - root需要是根目录，目录下有几个文件夹，每个文件夹表示一个类别
    - transform和target_tansform 是图片增强
    - loader是图片读取办法，将图片转换成我们需要的图片类型进入神经网络

- nn.Module(模组）

  > PyTorch里编写神经网络，所有的层结构和损失函数都来自于torch.nn,所有的模型构建都是从这个基类nn.Module继承的，于是有了以下模板-

  ```python
  class net_name(nn.Module):
      def __init__(self,other_arguments):
          super(net_name,self).__init__()
          self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size)
          # other network layer
      def forward(self,x):
          x = self.conv1(x)
          return x
  ```

  这样就建立了一个计算图，并且结构可以复用多次，每次调用就相当于用该计算图定义的相同参数做一次前向传播，这得益于PyTorch的自动求导功能，不需要自己编写反向传播

  - 定义完模型后我们需要nn这个包来定义损失函数，常见的损失函数已经定义在了nn中，比如：均方误差，多分类的交叉熵，二分类的交叉熵等

  - 调用定义好的损失函数也很简单：

    ```python
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output,target)
    ```

- torch.optim (优化）

  > 在机器学习或深度学习中，我们需要通过修改参数使得损失函数最小化或最大化，优化算法是一种调参策略

  1. 一阶优化算法

  2. 二阶优化算法

     pg30

  

  

  

  

  

  

  

  

  

  

  

  

  

  