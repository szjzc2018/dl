## 1. Discriminative Model

### 1.1 Structure & Loss Function

回顾我们的目标,我们希望找到 $NN_{\theta}$ ,使得 $NN_{\theta}(x) \approx f(x) (x\in X_0)$ , 而我们有一些训练数据 $P:\{(x,y)\}$ ,满足 $y=f(x)$ . 于是，我们先训练 $NN_{\theta}$ ,使得 $NN_{\theta}(x) \approx f(x) (x\in P)$ . 然而，对于离散的label,我们很难定义一个比较好的可以求导的损失函数，于是我们引入一个重要的思想:让我们的 $NN_{\theta}$ 对每个标签生成一个概率 $p_{\theta,y}(x)$ ，而我们的目的则变为找到 $\theta$ 使得 $p_{\theta,f(x)}(x)$ 尽可能接近 $1$ .

于是，这就引出了我们的损失函数的定义:交叉熵(cross-entropy)损失函数:

$$
L(\theta)=-\frac{1}{|P|}\sum_{(x,y)\in P}\log p_{\theta,y}(x)
$$

(对于一般的两个概率分布 $p,q$ ,交叉熵定义为 $H(p,q)=-\sum_x p(x)\log q(x)$ ,而上面的损失函数就是 $p$ 与一个 $f(x)$ 处单点分布的交叉熵)

### 1.2 Layers

说了这么多， $NN$ 的结构究竟是什么样？事实上， $NN$ 的结构是非常灵活（你也可以发明你想要的结构！），这里我们介绍一些常见的结构和它们的特点

#### 1.2.1 Fully Connected Layer & Activation Function

最常见的结构就是全连接层，它的数学表达式可以写成:

$$
f: \mathbb{R}^n \rightarrow \mathbb{R}^m, f(x)=\sigma(Wx+b)
$$

其中 $W \in \mathbb{R}^{m \times n}, b \in \mathbb{R}^m$ . 全连接层十分灵活，并且具有很强的表达能力（拟合各种函数的能力），同时可以很方便地进行维度的转换，许多简单的判别任务仅通过全连接层（和马上要说的）激活函数就可以达到比较令人满意的效果。

激活函数 $\sigma$ 是全连接层的重要组成部分，很容易证明，如果没有激活函数的话，事实上多少个全连接层复合都和一个没有区别，并且最终得到的是一个线性的函数，这大多数时候是不够的。(大多数判别任务并没有线性的特征！例如数字识别，把两个0的图片加起来取平均，得到的标签可能是8)常见的激活函数有 $sigmoid(x) = \frac{1}{1+e^{-x}}, tanh(x)= \frac{e^x-e^{-x}}{e^x+e^{-x}}, ReLU(x)=\max(0,x)$ 等等，在不同的任务中，不同的激活函数可能会有不同的效果。

#### 1.2.2 Convolutional Layer

全连接层表达能力非常强，但是，它也存在两个缺点：首先，由于向量的维度一般很大，所以参数量也很大，导致训练时间长；其次，全连接层没有用到数据的任何特点，而是只看成一个向量，这可能会导致过拟合的问题，并且也会浪费我们所知道的关于 $f$ 的某些信息
所以，对于图片类型的数据（这也是深度学习主要应用之一），我们想要引入一个新的结构，这就是卷积层。

卷积层的intuition来自两个观察:首先，对图片， $f$ 具有一个很好的性质,那就是平移不变性（想象把一张数字2平移1个像素，它仍然是2,并且人眼几乎看不出区别，但如果转化成向量以后，这两个向量也许完全不同）；其次，来自于人判断图片的方式：根据局部的特征来判断图片的内容。于是，我们引入卷积层来获得图片的局部特征。

卷积层的数学表达式可以写成:

$$
f: \mathbb{R}^{C_1\times n\times m} \to \mathbb{R}^{C_2\times n'\times m'} 
$$

 

$$
f(x)_{c_2,i,j} = \sum_{k=1}^{k_1}\sum_{l=1}^{k_2}W_{c_1,c_2,k,l}x_{c_1,i+k,j+l}+b_{c_1,c_2,k,l}
$$

这看上去很恐怖，但实际上表达的意思就是，我们用 $k_1\times k_2$ 的方框来“扫描”这个图片的每个区域，然后通过一个线性函数（称为卷积核）得到这个区域所对应的值，从而得到新的“图片”，称为特征图。

为什么卷积可以提取局部特征？我们可以用一个简单的例子来直观感受一下，假如我们想确定一张图片（每个像素范围在[0,1]内）中有没有一条长为3的竖线，形如（ $\begin{bmatrix}0&1&0\\0&1&0\\0&1&0\end{bmatrix}$ ），我们可以用一个 $3\times 3$ 的卷积核 $\begin{bmatrix}-1&1&-1\\-1&1&-1\\-1&1&-1\end{bmatrix}$ 来扫描这个图片，如果这个图片中有一条竖线，那么最终得到的特征图中，这个位置所对应的值就应该是3,否则不可能达到3.这就是卷积层提取局部特征的原理，特征图中的每个值都对应着原来图片中的一个局部的特征。

到这里，我们也可以很自然的引入池化层的概念：比如在上面判别竖线的例子中，只有很大的值代表我们提取到的特征，而对于不大不小的值，事实上没有特别大的意义，所以我们可以用最大池化的方式，保留特征图的每个 $K\times K$ 中的最大值。（一个夸张的例子就是在上面的例子中，如果对输出的整个特征图做最大池化，那么最终的输出就是一个 $1\times 1$ 的特征图，如果是3，那么就是有竖线，否则没有）

卷积层（池化层）的优点是参数量小（想象在上面寻找竖线的例子里，如果要用全连接层，我们至少要图片维度大小量级的参数！但是卷积层则只需要9个），可以很好地利用图片的局部特征，而且也可以很好地处理平移不变性的问题。对于卷积层输出的每个值，在原来的图片上和这个值相关的区域成为感受野(receptive field).

PS: 在卷积的“扫描”过程中，我们每次移动的步长也可以不为1,这样就可以起到降维的作用，步长被称为stride.一般地说，如果原图大小为 $n\times m$ ,卷积核大小为 $k_1\times k_2$ ,stride为 $s_1\times s_2$ ,那么输出的特征图大小为 $\lfloor \frac{n-k_1}{s_1} \rfloor \times \lfloor \frac{m-k_2}{s_2} \rfloor$ . 为了调整输出的大小，我们可以在原图的周围加一圈0，这被称为padding. 如果padding的大小为 $p_1\times p_2$ （ $p_1,p_2$ 分别是上下,左右的padding行/列数）
,那么输出的特征图大小为 $\lfloor \frac{n-k_1+2p_1}{s_1} \rfloor \times \lfloor \frac{m-k_2+2p_2}{s_2} \rfloor$ .

其中in_channels是输入的通道数（特征图的个数），out_channels是输出的通道数，kernel_size是卷积核的大小，stride是步长，padding是padding的大小。

### 1.2.3 Batchnorm Layer

Batchnorm是一种用来稳定训练的层，它和pooling层一样没有可训练参数，作用就是把每一层的输出都归一化到均值为0，方差为1的分布，这样可以使得每一层的输出都在一个比较稳定的范围内，从而避免数值过大或者过小导致的梯度消失或者爆炸问题。

在pytorch中，batchnorm层可以用如下的函数实现：

```python
torch.nn.BatchNorm2d(num_features)
```

其中num_features是输入的通道数。

### 1.3 Prevent Overfitting

#### 1.3.1 Regularization & Data Augmentation

还记得我们在前面提到的过拟合问题吗?

一般地说，我们只能训练 $NN$ ,使得 $NN(x) \approx f(x) (x\in P)$ , 而我们希望 $NN(x) \approx f(x) (x\in X_0)$ . 这里当然要假定 $X_0$ 有一些比较好的性质，否则这两者之间可能毫不相关。由于 $X_0$ 本身就没有数学上的严格定义，它表示我们实际需要判别的图片，那么，我们可以来思考一下， $X_0$ 和 $P$ 之间可能有什么关系?

一个很自然的想法就是 $X_0$ 和 $P$ 应该比较接近，从而，如果我们的 $f$ 在比较接近的数据上表现不会差很多的话，效果应该比较好。然而，这一点并不好刻画，我们转而限制一个与之正相关的量：模型的参数大小，这多少蕴含了模型梯度的大小，于是可以起到一定的作用。具体地，我们在损失函数中加入一个正则项，这个正则项一般是参数的范数，这样可以使得参数不会过大，从而避免过拟合。根据加入正则项的不同，我们可以分为L1正则和L2正则，分别是参数的绝对值和平方的和。而这个正则项的系数是一个人为设定的超参数，称为正则化系数(weight decay)。

除了对 $f$ 加以限制以外，我们还有另一个自然的想法：根据实际的情况自己扩展 $P$ !因为 $X_0$ 本来就是基于某些主观特征和性质定义的东西，所以我们可以直接根据这些性质来生成一些数据拓展 $P$ ,从而让模型在 $P$ 和 $X_0$ 的表现上更一致。例如，在识别数字中，我们知道一个数字随机加一点点小的扰动还是这个数字，于是我们可以把训练的数据加上一点噪声，标签不变，成为新的训练数据。这个过程被称为数据增强(Data Augmentation),常见的数据增强方法有旋转，翻转，缩放等等。

#### 1.3.2 Dropout

Dropout也是一种用来防止过拟合的方法，它主要是为了避免一些偶然出现但不具备generalization能力的非常强的特征。它的思想很简单：在训练的时候，我们随机地让一些神经元失活，也就是说，让它们的输出为0，这样可以使得模型不会过分依赖某些神经元，从而避免过拟合。

#### 1.3.3 Early Stopping

还有一个耍赖的方法:早停(Early Stopping)。它的思想也很简单，既然在 $P$ 上的表现和在 $X_0$ 上表现的关系我们弄不清，而事实上我们最终是用测试集来表示 $X_0$ 得到表现，那我们直接把训练集划出一部分作为验证集，然后在验证集上表现不再提升的时候停止训练。这在最终测试集和训练集的分布比较一致的时候是有效的，但是如果训练集和测试集的分布不一致，那么早停可能会导致模型在测试集上表现不好。

### 1.4 Structure

#### 1.4.1 CNN

CNN 的结构是卷积层和池化层的交替，最后加上全连接层，这样的结构可以很好地提取图片的局部特征，而且也可以很好地处理平移不变性的问题。CNN的结构是深度学习中最常见的结构，也是最有效的结构之一。

#### 1.4.2 ResNet

在CNN中，我们可以很容易地发现一个问题：随着网络的加深，梯度消失和爆炸问题会变得越来越严重，从而导致训练困难。ResNet的提出就是为了解决这个问题，它的核心思想是引入了一个shortcut，也就是说，我们不是直接把输入传到输出，而是把输入和输出相加，这样可以使得梯度更容易传播，从而可以训练更深的网络。具体地说，一个ResBlock的结构是这样的:

$$
f(x)=x+C(x)
$$

其中 $C(x)$ 是一个卷积层（一般由两个卷积层组成），这样，我们就可以很容易地训练一个很深的网络，而且也可以很好地处理梯度消失和爆炸问题。

#### 1.4.3 DenseNet

DenseNet是另一种解决梯度消失和爆炸问题的方法，它的核心思想是引入了一个dense block，也就是说，我们不是把输入和输出相加，而是把输入和输出拼接在一起，这样可以使得梯度更容易传播，从而可以训练更深的网络。具体地说，一个DenseBlock的结构是这样的:

$$
x_0 = input
$$

$$
x_i = concat(x_{i-1},C_{i-1}(x_{i-1}))
$$

其中 $C$ 是卷积层， $concat$ 是拼接操作，在每一次操作后，可以发现特征图的个数都在增加，于是到若干层以后，我们使用一个transition layer来减少特征图的个数，这就构成了DenseNet的结构。

> Densenet 是CVPR17的best paper, 是姚班学长做出来的，但是颁奖的时候他们在外面旅游导致没人领奖，场面一度尴尬.(Yi Wu)

### 1.5 Code Implementation

接下来，我们展示一些基于Pytorch的简单代码实现，由于作者水平有限，如果代码哪里有bug请不要骂我.

全连接神经网络:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

卷积神经网络：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 9216)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

ResNet:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        return x+x2

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.res1 = ResBlock(32, 32)
        self.res2 = ResBlock(32, 32)
        self.fc = nn.Linear(28*28*32, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.res1(x)
        x = self.res2(x)
        x = x.view(-1, 28*28*32)
        x = self.fc(x)
        return x
```

DenseNet:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels+out_channels, out_channels, 3, 1, 1)
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(torch.cat([x, x1], 1)))
        return torch.cat([x, x1, x2], 1)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.pool = nn.AvgPool2d(2, 2)
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        return x

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.dense1 = DenseBlock(32, 32)
        self.trans1 = TransitionLayer(96, 64)
        self.dense2 = DenseBlock(64, 32)
        self.trans2 = TransitionLayer(96, 64)
        self.fc = nn.Linear(7*7*64, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = x.view(-1, 7*7*64)
        x = self.fc(x)
        return x
```
