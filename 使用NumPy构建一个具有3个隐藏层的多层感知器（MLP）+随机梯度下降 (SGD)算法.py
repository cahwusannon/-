# 使用NumPy构建一个具有3个隐藏层的多层感知器（MLP）来进行红酒品质分类。
import numpy as np  # 该库用于进行数据处理（包括数据转换以及计算）
import pandas as pd  # 导入红酒txt文件数据的包

file_path = 'D:\edge浏览器\大三上\智能系统\智能系统考试2023\数据集\wine\wine.data'  # 读取数据
data = pd.read_csv(file_path,   #  # pd.read_csv(filename, ...):这是pandas库的一个函数，用于从CSV文件中读取数据。filename是文件的路径，...是一系列任选参数
                   names=['class', 'Al', 'Ma', 'Ash', 'Aoa', 'Mag',
                                     'Top', 'Fl', 'No', 'Pr', 'Co', 'Hue',
                                     'OD', 'Pro'])   # 读取CSV文件，将每一列的名称指定为提供的名称列表

# 数据划分
x = data.drop('class', axis=1)  # 从数据集中移除名为“class”的列，并提取剩余列为特征
y = data['class']  # 数据集中名为“class”的列是红酒品质类别的标签
train_indices = list(range(0, 30)) + list(range(59, 95)) + list(range(130, 153))  # 定义划分索引 训练集
test_indices = list(range(30, 59)) + list(range(95, 130)) + list(range(153, 178))   # 定义划分索引 测试集

# 根据索引划分数据集,并从整体数据集中提取了相应集的红酒品质类别class标签
X_train, X_test, y_train, y_test = x.iloc[train_indices], x.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]  # iloc根据索引的使用方法选择相应的行

# -------------------------------------------------------------------------------------------------------------------
# 定义了两个常用的神经网络激活函数：ReLU（Rectified Linear Unit）和Softmax,ReLU 用于隐藏层，提供非线性映射，使网络能够学习复杂的函数。Softmax 通常用于多类别分类问题的输出层，将模型的原始输出转换为概率分布，便于解释和分类
def relu(x):    # 输入x是一个数值或数组
    return np.maximum(0, x)  # 返回输入元素和零之间的较大值,实现了一个简单的非线性映射，对于正数输入有激活作用，对于负数输入则不激活（输出为零
def softmax(x):   # 输入x是一个包含类别分数的数组，通常是神经网络输出的原始分数
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))    # 减去 np.max(x, axis=1, keepdims=True) 是为了数值稳定性，防止指数爆炸。这一操作不改变 Softmax 结果，但可以减小计算过程中的数值范围
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)     # np.sum(exp_x, axis=1, keepdims=True) 对每个样本的指数运算结果进行求和。最后的结果是将每个类别分数转换为对应类别的概率


# 随机梯度下降 (SGD)算法 定义一个3层感知器（MLP）神经网络模型，并提供了网络的初始化和前向传播方法
# 在神经网络模型类中添加中间层的输出属性
class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):  # 初始化方法 __init__，在创建神经网络对象时调用。这段代码的目的是初始化神经网络的参数，包括权重和偏置，并添加中间层的输出属性
        self.weights_input_hidden1 = np.random.randn(input_size, hidden_size1)  # 输入层到第一个隐藏层的权重矩阵，通过 np.random.randn 生成随机数初始化。这个权重矩阵的大小为 (input_size, hidden_size1)，其中 input_size 是输入层的大小，hidden_size1 是第一个隐藏层的大小
        self.bias_input_hidden1 = np.zeros((1, hidden_size1))  # 第一个隐藏层的偏置，通过 np.zeros 生成全零的数组初始化。这个偏置的大小为 (1, hidden_size1)
        self.weights_hidden1_hidden2 = np.random.randn(hidden_size1, hidden_size2)  # 第一个隐藏层到第二个隐藏层的权重矩阵，同样通过 np.random.randn 生成随机数初始化。这个权重矩阵的大小为 (hidden_size1, hidden_size2)
        self.bias_hidden1_hidden2 = np.zeros((1, hidden_size2))  # 第二个隐藏层的偏置，通过 np.zeros 生成全零的数组初始化。这个偏置的大小为 (1, hidden_size2)
        self.weights_hidden2_hidden3 = np.random.randn(hidden_size2, hidden_size3)  # 第二个隐藏层到第三个隐藏层的权重矩阵，同样通过 np.random.randn 生成随机数初始化。这个权重矩阵的大小为 (hidden_size2, hidden_size3)
        self.bias_hidden2_hidden3 = np.zeros((1, hidden_size3))  # 第三个隐藏层的偏置，通过 np.zeros 生成全零的数组初始化。这个偏置的大小为 (1, hidden_size3)
        self.weights_hidden3_output = np.random.randn(hidden_size3, output_size)  # 在神经网络模型中初始化第三个隐藏层到输出层之间的权重矩阵。这个矩阵中的权重值将在训练过程中通过反向传播进行调整，以使得神经网络能够更好地学习输入数据的表示
        self.bias_hidden3_output = np.zeros((1, output_size))   # 输出层的偏置，通过 np.zeros 生成全零的数组初始化。这个偏置的大小为 (1, output_size)

        # 添加中间层的输出属性，初始化为 None。在前向传播的过程中，这两个属性将被用来存储中间层的输出
        self.hidden_output1 = None
        self.hidden_output2 = None
        self.hidden_output3 = None

    # 在 forward 方法中
    def forward(self, x):
        # 输入层到第一个隐藏层
        hidden_input1 = np.dot(x, self.weights_input_hidden1) + self.bias_input_hidden1  # 第一个隐藏层的输入，通过将输入 x 与第一个隐藏层的权重矩阵相乘，再加上第一个隐藏层的偏置得到。这个操作用 np.dot 实现

        self.hidden_output1 = relu(hidden_input1)   # 第一个隐藏层的输出，通过将 hidden_input1 应用激活函数 relu 得到

        # 第一个隐藏层到第二个隐藏层
        hidden_input2 = np.dot(self.hidden_output1, self.weights_hidden1_hidden2) + self.bias_hidden1_hidden2   # 第二个隐藏层的输入，通过将第一个隐藏层的输出 self.hidden_output1 与第二个隐藏层的权重矩阵相乘，再加上第二个隐藏层的偏置得到
        self.hidden_output2 = relu(hidden_input2)   # 第二个隐藏层的输出，通过将 hidden_input2 应用激活函数 relu 得到

        # 第二个隐藏层到第三个隐藏层
        hidden_input3 = np.dot(self.hidden_output2, self.weights_hidden2_hidden3) + self.bias_hidden2_hidden3  # 第三个隐藏层的输入，通过将第二个隐藏层的输出 self.hidden_output2 与第三个隐藏层的权重矩阵相乘，再加上第三个隐藏层的偏置得到
        self.hidden_output3 = relu(hidden_input3)   # 第三个隐藏层的输出，通过将 hidden_input3 应用激活函数 relu 得到

        # 第三个隐藏层到输出层
        output_input = np.dot(self.hidden_output3, self.weights_hidden3_output) + self.bias_hidden3_output  # 输出层的输入，通过将第三个隐藏层的输出 self.hidden_output3 与输出层的权重矩阵相乘，再加上输出层的偏置得到
        output = softmax(output_input)  # 神经网络的输出，通过将 output_input 应用激活函数 softmax 得到。在多类别分类问题中，softmax 函数常用于将输出转化为类别概率

        return output   # 返回output， 即神经网络对输入数据的预测输出。这个输出可以与实际标签进行比较，用来计算损失并进行反向传播，以更新神经网络的参数


# -------------------------------------------------------------------------------------------------------------------
# 数据预处理（使用标准化），有助于提高模型的训练效果，特别是在涉及到梯度下降等优化算法时。标准化可以确保模型更容易学到合适的权重。 NumPy 数组的使用有助于与神经网络模型的输入和输出兼容
mean = X_train.mean(axis=0)  # 计算训练集中每个特征的均值。axis=0 表示沿着列的方向计算均值
std = X_train.std(axis=0)   #  计算训练集中每个特征的标准差。axis=0 表示沿着列的方向计算标准差
X_train_normalized = (X_train - mean) / std  #  对训练集进行标准化处理。将每个特征减去均值，然后除以标准差。这一步可以帮助确保不同特征的数值范围差异不会对模型产生不良影响
X_test_normalized = (X_test - mean) / std   # 对测试集进行与训练集相同的标准化处理。这里使用训练集的均值和标准差来标准化测试集，以保持一致性

# 转换为NumPy数组
X_train_array = X_train_normalized.values   #  将训练集数据转换为NumPy数组。.values 将 pandas DataFrame 转换为 NumPy 数组
y_train_array = y_train.values   # 将训练集标签转换为NumPy数组
X_test_array = X_test_normalized.values  # 将测试集数据转换为NumPy数组
y_test_array = y_test.values    # 将测试集标签转换为NumPy数组

# 创建并训练神经网络模型
input_size = X_train_array.shape[1]  # 获取训练数据的特征数，即输入层的大小。X_train_array 是训练数据集的特征矩阵，shape[1] 返回特征的数量
hidden_size1 = 128  # 定义第一个隐藏层的神经元数量，可以调整这个值来改变隐藏层的大小，从而影响神经网络的容量
hidden_size2 = 64  # 定义第二个隐藏层的神经元数量，与第一个隐藏层一样，可以调整这个值
hidden_size3 = 32  # 添加第三个隐藏层的神经元数量
output_size = len(np.unique(y_train_array))  # 获取训练数据的输出类别数，即输出层的大小。y_train_array 是训练数据集的标签，np.unique 用于获取唯一的类别，并用 len 获取类别的数量

model = NeuralNetwork(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)   # 创建神经网络模型的实例，使用上述定义的输入层大小、两个隐藏层的神经元数量和输出层大小

# 定义损失函数和学习率，学习率是一个超参数，它决定了模型在每次迭代中更新权重的步长。过大的学习率可能导致模型不稳定，而过小的学习率可能导致模型收敛速度过慢。这个值可以根据训练过程的效果进行调整
learning_rate = 0.0001

# -训练模型------------------------------------------------------------------------------------------------------------------
# 神经网络的训练过程，使用随机梯度下降（Stochastic Gradient Descent, SGD）进行权重和偏置的更新
epochs = 1000   # 定义训练的迭代次数，即模型将遍历整个训练数据集的次数
for epoch in range(epochs):  #  开始迭代训练过程
    # 前向传播
    predictions = model.forward(X_train_array)  # 计算模型的预测结果
    one_hot_targets = np.eye(output_size)[y_train_array - 1]   # 将训练数据的标签转换为独热编码。np.eye(output_size) 生成一个单位矩阵，通过索引 y_train_array - 1 转换为独热编码

    # 反向传播
    output_error = predictions - one_hot_targets    # 计算输出误差，即模型预测值与实际标签之间的差异
    hidden3_error = np.dot(output_error, model.weights_hidden3_output.T) * (model.hidden_output3 > 0)   # 计算第三个隐藏层的误差，通过反向传播误差从输出层传播到隐藏层
    hidden2_error = np.dot(hidden3_error, model.weights_hidden2_hidden3.T) * (model.hidden_output2 > 0)   # 计算第二个隐藏层的误差，通过反向传播误差从第三隐藏层传播到第二个隐藏层
    hidden1_error = np.dot(hidden2_error, model.weights_hidden1_hidden2.T) * (model.hidden_output1 > 0)  # 计算第一个隐藏层的误差，通过反向传播误差从第二隐藏层传播到第一个隐藏层

    # 更新权重和偏置
    model.weights_hidden3_output -= learning_rate * np.dot(model.hidden_output3.T, output_error)  #  更新第三个隐藏层到输出层的权重
    model.bias_hidden3_output -= learning_rate * np.sum(output_error, axis=0, keepdims=True)    # 更新第三个隐藏层到输出层的偏置
    model.weights_hidden2_hidden3 -= learning_rate * np.dot(model.hidden_output2.T, hidden3_error)   # 更新第二个隐藏层到第三个隐藏层的权重
    model.bias_hidden2_hidden3 -= learning_rate * np.sum(hidden3_error, axis=0, keepdims=True)  # 更新第二个隐藏层到第三个隐藏层的偏置
    model.weights_hidden1_hidden2 -= learning_rate * np.dot(model.hidden_output1.T, hidden2_error)  # 更新第一个隐藏层到第二个隐藏层的权重
    model.bias_hidden1_hidden2 -= learning_rate * np.sum(hidden2_error, axis=0, keepdims=True)  # 更新第一个隐藏层到第二个隐藏层的偏置
    model.weights_input_hidden1 -= learning_rate * np.dot(X_train_array.T, hidden1_error)   # 更新输入层到第一个隐藏层的权重
    model.bias_input_hidden1 -= learning_rate * np.sum(hidden1_error, axis=0, keepdims=True)  # 更新输入层到第一个隐藏层的偏置

    # 计算损失函数（交叉熵）
    epsilon = 1e-8  # 定义一个很小的常数，用于防止出现除以零的情况
    predictions = np.clip(predictions, epsilon, 1 - epsilon)    # 对模型的预测值进行裁剪，确保在取对数时不会出现无效值
    loss = -np.sum(one_hot_targets * np.log(np.clip(predictions, epsilon, 1 - epsilon))) / len(y_train_array)   # 计算交叉熵损失

    if epoch % 100 == 0:    # 每100次迭代打印一次
        print(f'Epoch {epoch}, Loss: {loss}')  # 打印损失值，用于监控训练过程

# -------------------------------------------------------------------------------------------------------------------
# 在测试集上评估训练好的神经网络模型
test_predictions = model.forward(X_test_array)  #  对测试集进行前向传播，得到模型的预测结果

final_predictions = model.forward(X_test_array)   # 在循环外部获取最终的预测标签集合final_predictions
predicted_labels = np.argmax(final_predictions, axis=1) + 1  #  final_predictions 是模型的输出，对预测结果取最大值的索引+1，即得到模型预测的类别
print(predicted_labels)  # 输出预测结果
accuracy = np.mean(predicted_labels == y_test_array)    # : 计算模型在测试集上的准确率。将预测的类别与实际类别比较，然后计算平均准确率
print(f'Test accuracy: {accuracy}')  #  打印测试集上的准确率





