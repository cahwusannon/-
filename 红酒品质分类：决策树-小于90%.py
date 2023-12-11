import pandas as pd     # 导入 Pandas 库，用于数据处理
import numpy as np     # 导入 NumPy 库，用于数值计算
from sklearn.tree import DecisionTreeClassifier    # 导入 DecisionTreeClassifier 类，用于构建决策树分类器
from sklearn.metrics import accuracy_score  # 导入 accuracy_score 函数，用于计算分类器的准确性
from sklearn.model_selection import train_test_split    # 导入 train_test_split 函数，用于将数据集划分为训练集和测试集

# 读取数据
file_path = 'D:\edge浏览器\大三上\智能系统\智能系统考试2023\数据集\wine\wine.data'
data = pd.read_csv(file_path, names=['class', 'Al', 'Ma', 'Ash', 'Aoa', 'Mag',
                                     'Top', 'Fl', 'No', 'Pr', 'Co', 'Hue',
                                     'OD', 'Pro'])

# 数据划分
# 从数据中提取特征变量X 和目标变量y，并将数据集划分为训练集和测试集
X = data.drop('class', axis=1)  # 提取特征变量X，去掉 'class' 列
y = data['class']  # 提取目标变量y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   # 利用 train_test_split 将数据集划分为训练集和测试集，其中 test_size 表示测试集所占比例，random_state 用于设置随机种子以确保可重复性

# 遗传算法参数
# 定义遗传算法的相关参数
num_features = 13  # 13特征数量
chromosome_length = num_features  # 染色体长度等于特征数量
parameter_ranges = [(0.01, 1.0),  # 染色体中第一个基因的取值范围
                    (0.01, 0.5),  # 染色体中第二个基因的取值范围
                    (0.01, 0.3),  # ...
                    (0.01, 5.0),
                    (0.01, 30.0),
                    (0.01, 1.0),
                    (0.01, 0.2),
                    (0.01, 10.0),
                    (0.01, 2.0),
                    (0.01, 1.0),
                    (0.01, 0.25),
                    (0.01, 1.12),
                    (0.01, 500.0)
                    ]
population_size = 50  # 种群大小
num_generations = 50  # 遗传算法的迭代次数
crossover_rate = 0.8  # 交叉概率，决定父代基因的哪一部分进行交叉
mutation_rate = 0.1  # 变异概率，决定染色体是否发生变异


# 遗传算法适应度函数
def fitness_function(chromosome):
    # 将染色体中的整数值转换为浮点数
    chromosome = [float(value) for value in chromosome]

    # 将染色体中的值映射到具体的参数
    hyperparameters = {
        'criterion': 'entropy' if chromosome[0] < 0.5 else 'gini',
        'splitter': 'random' if chromosome[1] < 0.5 else 'best',
        'min_samples_split': int(chromosome[2] * 10) + 2,
        'min_samples_leaf': int(chromosome[3] * 10) + 1,
        'max_features': 'sqrt' if chromosome[4] < 0.5 else None,
        'max_depth': None if chromosome[5] < 0.5 else int(chromosome[6] * 10) + 1,
        'min_impurity_decrease': chromosome[7],
        'random_state': 42
    }

    model = DecisionTreeClassifier(**hyperparameters)   # 创建决策树模型，使用定义的超参数
    model.fit(X_train, y_train)     # 在训练集上拟合模型
    y_pred = model.predict(X_test)  # 在测试集上进行预测

    accuracy = accuracy_score(y_test, y_pred)  # 计算分类器的准确性作为适应度值
    return accuracy     # 返回准确度值

# 初始化种群
def initialize_population(population_size, chromosome_length):
    population = np.random.rand(population_size, chromosome_length)  # 生成随机的种群，表示初始的种群，每个染色体的基因包含多个基因且取值在相应范围内

    for ii in range(chromosome_length):     # 循环遍历每个基因的索引，将基因的取值范围映射到具体的参数范围
        (min_value, max_value) = parameter_ranges[ii]   # 获取当前基因的取值范围
        population[:, ii] = population[:, ii] * (max_value - min_value) + min_value  # 对每个染色体的当前基因进行映射，确保在合适的范围内

    return population    # 返回初始化的种群


# 遗传算法主循环
# 利用初始化的种群进行遗传算法的主循环
population = initialize_population(population_size, chromosome_length)
best_chromosomes = []  # 用于存储每一代的最佳染色体

# 迭代遗传算法的代数
for generation in range(num_generations):   # 循环进化
    print(f"Generation {generation + 1}:")   # 打印当前进化的代数
    fitness_values = [fitness_function(chromosome) for chromosome in population]  # 计算种群中每个染色体的适应度值

    # 选取适应度值最高的染色体索引，即选择最优个体
    selected_indices = np.argsort(fitness_values)[-population_size:]    # 通过 np.argsort 对适应度值进行排序，然后取排序后的后 population_size 个索引，即选取适应度值最高的前 population_size 个染色体的索引。这样就得到了当前代中适应度值最高的染色体的索引列表，存储在 selected_indices 中
    best_chromosome = population[selected_indices[-1]].copy()   # 从当前代的种群中选择适应度值最高的染色体，即 selected_indices 中最后一个索引对应的染色体。copy() 操作用于创建染色体的深拷贝，确保后续对 best_chromosome 的修改不会影响原始种群。这个最优染色体将被记录并在后续的遗传算法迭代中继续使用
    best_chromosomes.append(best_chromosome)  # 记录每一代的最佳染色体

    # 交叉操作：以一定概率对种群中的染色体进行交叉
    for i in range(0, population_size, 2):  #  循环对于种群中的每两个相邻的染色体进行判断是否交叉操作
        if np.random.rand() < crossover_rate:  # 如果随机数小于交叉率 crossover_rate，则进行交叉操作随机选择一个交叉点之后的基因进行交换，以增加遗传算法的多样性
            crossover_point = np.random.randint(chromosome_length)   # 生成一个随机的交叉点，决定从哪里开始进行交叉
            # 将两个染色体的基因进行交叉
            population[selected_indices[i], crossover_point:] = population[selected_indices[i + 1], crossover_point:]
            population[selected_indices[i + 1], crossover_point:] = population[selected_indices[i], crossover_point:]

    # 变异操作：以一定概率对种群中的染色体进行变异
    for i in range(population_size):  # 循环对于当前种群来进行判断是否要变异操作
        if np.random.rand() < mutation_rate:  # 对于每个染色体，如果随机数小于变异率 mutation_rate，则进行变异操作
            mutation_point = np.random.randint(chromosome_length)   # 生成一个随机的变异点，决定对染色体的哪个基因进行变异
            min_val, max_val = parameter_ranges[mutation_point]  # 获取变异点的取值范围
            population[selected_indices[i], mutation_point] = np.random.uniform(min_val, max_val)   # 对染色体的变异点进行随机赋值，引入新的基因组合

    # 替换最不适应的染色体，保留最佳染色体
    min_fitness_index = np.argmin(fitness_values)  # 找到适应度值最小的染色体的索引
    population[min_fitness_index] = best_chromosome  # 将最不适应的染色体替换为上一代中的最佳染色体，确保种群中的染色体保持一定的多样性，并保留适应度值相对较高的个体

    best_fitness = max(fitness_values)   # 输出当前代数中的最佳适应度值
    print(f"Best Fitness: {best_fitness}")  # 输出当前进化中最好的适应度


# 获取最终代的最佳染色体
best_final_chromosome = best_chromosomes[-1]

# 创建最终模型
best_final_model = DecisionTreeClassifier(
    criterion='entropy' if best_final_chromosome[0] < 0.5 else 'gini',
    splitter='random' if best_final_chromosome[1] < 0.5 else 'best',
    min_samples_split=int(best_final_chromosome[2] * 10) + 2,
    min_samples_leaf=int(best_final_chromosome[3] * 10) + 1,
    max_features='sqrt' if best_final_chromosome[4] < 0.5 else None,
    max_depth=None if best_final_chromosome[5] < 0.5 else int(best_final_chromosome[6] * 10) + 1,
    min_impurity_decrease=best_final_chromosome[7],
    random_state=42
)

# 训练最终模型
best_final_model.fit(X_train, y_train)

y_pred_final = best_final_model.predict(X_test)  # 在测试集上评估最终模型
accuracy_final = accuracy_score(y_test, y_pred_final)   # 最终模型的预测准确率
print("Final Model Accuracy on Test Set:", accuracy_final)  # 输出最终模型的预测准确率
