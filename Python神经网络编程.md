# Python神经网络编程
> [blog](https://makeyourownneuralnetwork.blogspot.com/), [Github](https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork)
> [勘误](https://www.epubit.com/book/detail/34292), [douban](https://book.douban.com/subject/30192800/)

## 神经网络如何工作

神经网络每一**层**对应一个**权重矩阵**，矩阵的每一**行或列**对应一个**结点**。

反向传播误差时按**权重比例**分割误差，如 $errors_{hidden}=weight^T_{hidden-output}\cdot errors_{output}$ （实际计算时为简化计算，会忽略分母归一化因子，因为即使过大或过小，后续迭代网络会自行纠正）。
每一个隐层结点的误差是与该结点前向连接所有链接分割误差之和。


梯度下降避免跳过最小点，可通过调节步长（与梯度大小成反比）或选择不同起始参数（从多个不同起点）开始训练的方式。

误差函数的选择

 - 平滑连续
 - 越接近最小值，梯度越小

初始权重选择的经验法则：在 “$\pm$结点传入链接数量平方根倒数”的范围内随机采样，初始化权重。

**权重调整公式**
设 $w_{jk}$ 表示 $j$ 层到 $k$ 层权重，则
$$
\frac{\partial E}{\partial w_{jk}} = -(e_k)\cdot sigmoid(\sum_jw_{jk}\cdot o_j)(1-sigmoid(\sum_jw_{jk}\cdot o_j))\cdot o_j
$$
其中 $e_k=(t_k-o_k)$ 表示误差“目标值-实际值”；$sigmoid$ 中的求和公式表示进入最后一层结点的信号（应用激活函数之前）；$o_j$ 表示隐藏结点 $j$ 的输出。

加入学习因子$\alpha$（负号表示权重改变方向与梯度方向相反），有：
$$
new\space w_{jk}=old\space w_{jk}-\alpha\cdot\frac{\partial E}{\partial w_{jk}}
$$
可得矩阵形式：$$
\Delta W_{jk}=\alpha\cdot E_k\cdot sigmoid(O_k)(1-sigmoid(O_k))\cdot O_j^T
$$

## 使用Python进行DIY
> [完整代码见：neural_network_mnist_data.ipynb](https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork/blob/master/part2_neural_network_mnist_data.ipynb)

```python
class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # w_{input, hidden}
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        # w_{hidden, output}
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        self.lr = learningrate
        # expit(x) = 1/(1+exp(-x))
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    
    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        # 分割误差
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # 权重更新矩阵公式。transpose() 是转置
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        pass
```

## 实践


