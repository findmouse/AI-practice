# coding:utf-8
import os

# 关键：加入这一行，允许重复初始化 OpenMP 库
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import json
import torch
# 导入nn准备构建模型
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
# 导入torch的数据源 数据迭代器工具包
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
# 用于获得觉字母及字符规范化
import string
# 导入时间工具包
import time
# 引入制图工具包
import matplotlib.pyplot as plt

# 获取常见字符串


letters = string.ascii_letters + " ,.;'"
# print(letters)
n_letters = len(letters)

# 国家名 种类数
categorys = ['Italian', 'English', 'Arabic', 'Spanish', 'Scottish', 'Irish', 'Chinese', 'Vietnamese', 'Japanese',
             'French', 'Greek', 'Dutch', 'Korean', 'Polish', 'Portuguese', 'Russian', 'Czech', 'German']


def read_data(filename):
    my_list_x, my_list_y = [], []
    with open(filename, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            if len(line) <= 5:
                continue
                # 按照行提取样本x，样本y
            (x, y) = line.strip().split('\t')
            my_list_x.append(x)
            my_list_y.append(y)

    return my_list_x, my_list_y


class NameClassDataset(Dataset):
    def __init__(self, my_list_x, my_list_y):
        super().__init__()
        # 样本x
        self.my_list_x = my_list_x
        # 标签y
        self.my_list_y = my_list_y
        # 获取样本长度
        self.sample_len = len(my_list_x)

    # 定义魔法函数
    def __len__(self):
        return self.sample_len

    def __getitem__(self, index):
        # 一个一个地返回张量化后的单词和对应国家的编号
        # 对于Index异常值进行修正[0, self.sample_len-1]
        index = min(max(index, 0), self.sample_len - 1)
        # 按索引获数据样本x y
        x = self.my_list_x[index]
        y = self.my_list_y[index]

        # 样本one-hot张量化
        tensor_x = torch.zeros(len(x), n_letters)
        for index, letter in enumerate(x):
            tensor_x[index][letters.find(letter)] = 1

        tensor_y = torch.tensor(categorys.index(y), dtype=torch.long)
        # 返回编号index单词的的张量组合
        return tensor_x, tensor_y


# def test_dataset():
#     my_list_x, my_list_y = read_data('./data/name_classfication.txt')
#     my_dataset = NameClassDataset(my_list_x, my_list_y)
#     print(len(my_dataset))
#     print(my_dataset[0])
#     print(my_dataset[1])
#     print(my_dataset[2])


# 定义迭代器
def get_dataloader():
    my_list_x, my_list_y = read_data('./data/name_classfication.txt')
    my_dataset = NameClassDataset(my_list_x, my_list_y)
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
    print(f'len(my_dataloader)-->{len(my_dataloader)}')
    for tensor_x, tensor_y in my_dataloader:
        print(f"tensor_x.shape-->{tensor_x.shape}")
        print(f"tensor_y-->{tensor_y}")
        break


class My_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layer=1):
        super().__init__()
        # 词嵌入维度
        self.input_size = input_size
        # RNN模型输出的隐藏层维度
        self.hidden_size = hidden_size
        # 最终输出层单元个数
        self.output_size = output_size
        # RNN隐藏层个数
        self.num_layer = num_layer
        # 定义rnn层
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layer)

        # 定义全连接层
        self.linear = nn.Linear(self.hidden_size, self.output_size)

        # 定义softmax层, dim=-1以最后一维进行softmax计算
        self.softmax = nn.LogSoftmax(dim=-1)

    # 前向传播方法
    def forward(self, input, hidden):
        # input 输入的是二维的：shape:[6,57],代表6个单词，每个单词用57个数字表示
        input = input.unsqueeze(1)

        # 将input和hidden送往模型
        output, hn = self.rnn(input, hidden)
        # print(f'output-->{output.shape}')
        # print(f'output[-1]-->{output[-1].shape}')
        # print(f'hn-->{hn.shape}')
        result = self.linear(output[-1])

        # print(f'result.shape-->{result.shape}')
        # 经过 softmax
        return self.softmax(result), hn

    def initthidden(self):
        return torch.zeros(self.num_layer, 1, self.hidden_size)


# 测试模型

def test_model_RNN():
    # 实例化模型
    my_rnn = My_RNN(input_size=57, hidden_size=128, output_size=18)
    # 准备数据
    my_list_x, my_list_y = read_data('./data/name_classfication.txt')
    my_dataset = NameClassDataset(my_list_x, my_list_y)
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
    print(f'len(my_dataloader)-->{len(my_dataloader)}')
    for tensor_x, tensor_y in my_dataloader:
        input = tensor_x[0]
        hidden = my_rnn.initthidden()
        output, hn = my_rnn(input, hidden)
        print(f'output.shape-->{output.shape}')
        print(f'hn.shape-->{hn.shape}')
        # print(f"tensor_x.shape-->{tensor_x.shape}")
        # print(f"tensor_y-->{tensor_y}")
        break


class My_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layer=1):
        super().__init__()
        # 词嵌入维度
        self.input_size = input_size
        # RNN模型输出的隐藏层维度
        self.hidden_size = hidden_size
        # 最终输出层单元个数
        self.output_size = output_size
        # RNN隐藏层个数
        self.num_layer = num_layer
        # 定义rnn层
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layer)

        # 定义rnn层
        self.linear = nn.Linear(self.hidden_size, self.output_size)

        # 定义softmax层, dim=-1以最后一维进行softmax计算
        self.softmax = nn.LogSoftmax(dim=-1)

    # 前向传播方法
    def forward(self, input, hidden, c):
        # input 输入的是二维的：shape:[6,57],代表6个单词，每个单词用57个数字表示
        input = input.unsqueeze(1)

        # 将input和hidden送往模型
        output, (hn, cn) = self.rnn(input, (hidden, c))
        # print(f'output-->{output.shape}')
        # print(f'output[-1]-->{output[-1].shape}')
        # print(f'hn-->{hn.shape}')
        result = self.linear(output[-1])

        # print(f'result.shape-->{result.shape}')
        # 经过 softmax
        return self.softmax(result), hn, cn

    def initthidden(self):
        h0 = torch.zeros(self.num_layer, 1, self.hidden_size)
        c0 = torch.zeros(self.num_layer, 1, self.hidden_size)

        return h0, c0


def test_model_LSTM():
    # 实例化模型
    my_lstm = My_LSTM(input_size=57, hidden_size=128, output_size=18)
    # 准备数据
    my_list_x, my_list_y = read_data('./data/name_classfication.txt')
    my_dataset = NameClassDataset(my_list_x, my_list_y)
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
    print(f'len(my_dataloader)-->{len(my_dataloader)}')
    for tensor_x, tensor_y in my_dataloader:
        input = tensor_x[0]
        hidden, c = my_lstm.initthidden()
        output, hn, cn = my_lstm(input, hidden, c)
        print(f'output.shape-->{output.shape}')
        print(f'hn.shape-->{hn.shape}')
        print(f'cn.shape-->{cn.shape}')
        # print(f"tensor_x.shape-->{tensor_x.shape}")
        # print(f"tensor_y-->{tensor_y}")
        break


class My_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layer=1):
        super().__init__()
        # 词嵌入维度
        self.input_size = input_size
        # RNN模型输出的隐藏层维度
        self.hidden_size = hidden_size
        # 最终输出层单元个数
        self.output_size = output_size
        # RNN隐藏层个数
        self.num_layer = num_layer
        # 定义rnn层
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layer)

        # 定义rnn层
        self.linear = nn.Linear(self.hidden_size, self.output_size)

        # 定义softmax层, dim=-1以最后一维进行softmax计算
        self.softmax = nn.LogSoftmax(dim=-1)

    # 前向传播方法
    def forward(self, input, hidden):
        # input 输入的是二维的：shape:[6,57],代表6个单词，每个单词用57个数字表示
        input = input.unsqueeze(1)

        # 将input和hidden送往模型
        output, hn = self.gru(input, hidden)
        # print(f'output-->{output.shape}')
        # print(f'output[-1]-->{output[-1].shape}')
        # print(f'hn-->{hn.shape}')
        result = self.linear(output[-1])

        # print(f'result.shape-->{result.shape}')
        # 经过 softmax
        return self.softmax(result), hn

    def initthidden(self):
        return torch.zeros(self.num_layer, 1, self.hidden_size)


# 测试模型

def test_model_GRU():
    # 实例化模型
    my_gru = My_GRU(input_size=57, hidden_size=128, output_size=18)
    # 准备数据
    my_list_x, my_list_y = read_data('./data/name_classfication.txt')
    my_dataset = NameClassDataset(my_list_x, my_list_y)
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
    print(f'len(my_dataloader)-->{len(my_dataloader)}')
    for tensor_x, tensor_y in my_dataloader:
        input = tensor_x[0]
        hidden = my_gru.initthidden()
        output, hn = my_gru(input, hidden)
        print(f'output.shape-->{output.shape}')
        print(f'hn.shape-->{hn.shape}')
        # print(f"tensor_x.shape-->{tensor_x.shape}")
        # print(f"tensor_y-->{tensor_y}")
        break


my_lr = 1e-3
epochs = 1


# 训练rnn模型
def train_rnn():
    # 读取数据
    my_list_x, my_list_y = read_data('./data/name_classfication.txt')
    my_dataset = NameClassDataset(my_list_x, my_list_y)
    # 实例化模型
    # n_letters=57, hidden_size = 128, 类别总数output_size=18
    my_rnn = My_RNN(input_size=57, hidden_size=128, output_size=18)
    # 实例化损失函数对象
    my_null_loss = nn.NLLLoss()
    # 实例化优化器对象
    my_optim = optim.Adam(my_rnn.parameters(), lr=my_lr)

    # 定义打印日本函数
    start_time = time.time()
    total_iter_num = 0  # 已经训练样本的总数
    total_loss = 0  # 已经训练的损失
    total_loss_list = []  # 每隔n个样本，保存平均损失
    total_acc_num = 0  # 预测正确的样本个数
    total_acc_list = []  # 每隔n个样本，保存平均准确率
    # 开始训练
    for epoch_idx in range(epochs):
        # 实例化dataloader
        my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
        for i, (x, y) in enumerate(tqdm(my_dataloader)):
            # print(f"x.shape-->{x.shape}")
            # print(f"y-->{y}")
            output, hn = my_rnn(input=x[0], hidden=my_rnn.initthidden())
            # print(f'output.shape-->{output.shape}')  # [1, 18]
            # 计算损失
            my_loss = my_null_loss(output, y)
            # print(f'my_loss-->{my_loss}')
            # 梯度清零,因为是用优化器对梯度更新的，所以也要用优化器对梯度清零
            my_optim.zero_grad()
            # 反向传播
            my_loss.backward()
            # 梯度更新
            my_optim.step()

            # 统计下已经训练样本的总个数
            total_iter_num = total_iter_num + 1
            # 统计一下已经训练样本的总损失
            total_loss = total_loss + my_loss.item()
            # 统计已经训练的样本中预测正确的个数
            # a = torch.argmax(output)
            # print(f'a-->{a}')
            i_predict_num = 1 if torch.argmax(output).item() == y.item() else 0
            total_acc_num += i_predict_num
            # print(f"torch.argmax(output).item()-->{torch.argmax(output).item()}")
            # print(f"y-->{y}")
            # print(f'i_predict_num-->{i_predict_num}')

            # 每隔100次训练，保存一下平均损失和准确率
            if total_iter_num % 100 == 0:
                avg_loss = total_loss / total_iter_num
                total_loss_list.append(avg_loss)

                avg_acc = total_acc_num / total_iter_num
                total_acc_list.append(avg_acc)
            # 每隔2000次训练，打印一次日志
            if total_iter_num % 2000 == 0:
                temp_loss = total_loss / total_iter_num
                temp_acc = total_acc_num / total_iter_num
                temp_time = time.time() - start_time
                print('轮次：%d，损失：%.6f,时间：%d,准确率:%.3f' % (epoch_idx + 1, temp_loss, temp_time, temp_acc))
            pred = torch.argmax(output).item()

            # print(
            #     f"预测:{categorys[pred]} "
            #     f"真实:{categorys[y.item()]}"
            # )

        # 每个轮次保存模型
        torch.save(my_rnn.state_dict(), './save_model/my_rnn_model_%d.bin' % (epoch_idx + 1))
    # 计算总时间
    total_time = int(time.time() - start_time)
    # print(f'total_time-->{total_time}')
    save_data = {
        "loss": total_loss_list,
        "time": total_time,
        "acc": total_acc_list
    }

    with open('./save_results/ai_rnn_train_result.json', 'w', encoding='utf-8') as fw:
        json.dump(save_data, fw, ensure_ascii=False, indent=4)
    return total_loss_list, total_time, total_acc_list


# 训练rnn模型
def train_lstm():
    # 读取数据
    my_list_x, my_list_y = read_data('./data/name_classfication.txt')
    my_dataset = NameClassDataset(my_list_x, my_list_y)
    # 实例化模型
    # n_letters=57, hidden_size = 128, 类别总数output_size=18
    my_lstm = My_LSTM(input_size=57, hidden_size=128, output_size=18)
    # 实例化损失函数对象
    my_null_loss = nn.NLLLoss()
    # 实例化优化器对象
    my_optim = optim.Adam(my_lstm.parameters(), lr=my_lr)

    # 定义打印日本函数
    start_time = time.time()
    total_iter_num = 0  # 已经训练样本的总数
    total_loss = 0  # 已经训练的损失
    total_loss_list = []  # 每隔n个样本，保存平均损失
    total_acc_num = 0  # 预测正确的样本个数
    total_acc_list = []  # 每隔n个样本，保存平均准确率
    # 开始训练
    for epoch_idx in range(epochs):
        # 实例化dataloader
        my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
        for i, (x, y) in enumerate(tqdm(my_dataloader)):
            # print(f"x.shape-->{x.shape}")
            # print(f"y-->{y}")
            h0, c0 = my_lstm.initthidden()
            output, hn, cn = my_lstm(input=x[0], hidden=h0, c=c0)
            # print(f'output.shape-->{output.shape}')  # [1, 18]
            # 计算损失
            my_loss = my_null_loss(output, y)
            # print(f'my_loss-->{my_loss}')
            # 梯度清零,因为是用优化器对梯度更新的，所以也要用优化器对梯度清零
            my_optim.zero_grad()
            # 反向传播
            my_loss.backward()
            # 梯度更新
            my_optim.step()

            # 统计下已经训练样本的总个数
            total_iter_num = total_iter_num + 1
            # 统计一下已经训练样本的总损失
            total_loss = total_loss + my_loss.item()
            # 统计已经训练的样本中预测正确的个数
            # a = torch.argmax(output)
            # print(f'a-->{a}')
            i_predict_num = 1 if torch.argmax(output).item() == y.item() else 0
            total_acc_num += i_predict_num
            # print(f"torch.argmax(output).item()-->{torch.argmax(output).item()}")
            # print(f"y-->{y}")
            # print(f'i_predict_num-->{i_predict_num}')

            # 每隔100次训练，保存一下平均损失和准确率
            if total_iter_num % 100 == 0:
                avg_loss = total_loss / total_iter_num
                total_loss_list.append(avg_loss)

                avg_acc = total_acc_num / total_iter_num
                total_acc_list.append(avg_acc)
            # 每隔2000次训练，打印一次日志
            if total_iter_num % 2000 == 0:
                temp_loss = total_loss / total_iter_num
                temp_acc = total_acc_num / total_iter_num
                temp_time = time.time() - start_time
                print('轮次：%d，损失：%.6f,时间：%d,准确率:%.3f' % (epoch_idx + 1, temp_loss, temp_time, temp_acc))
            pred = torch.argmax(output).item()

            # print(
            #     f"预测:{categorys[pred]} "
            #     f"真实:{categorys[y.item()]}"
            # )

        # 每个轮次保存模型
        torch.save(my_lstm.state_dict(), './save_model/my_lstm_model_%d.bin' % (epoch_idx + 1))
    # 计算总时间
    total_time = int(time.time() - start_time)
    # print(f'total_time-->{total_time}')
    save_data = {
        "loss": total_loss_list,
        "time": total_time,
        "acc": total_acc_list
    }

    with open('./save_results/ai_lstm_train_result.json', 'w', encoding='utf-8') as fw:
        json.dump(save_data, fw, ensure_ascii=False, indent=4)
    return total_loss_list, total_time, total_acc_list


# 训练rnn模型
def train_gru():
    # 读取数据
    my_list_x, my_list_y = read_data('./data/name_classfication.txt')
    my_dataset = NameClassDataset(my_list_x, my_list_y)
    # 实例化模型
    # n_letters=57, hidden_size = 128, 类别总数output_size=18
    my_gru = My_RNN(input_size=57, hidden_size=128, output_size=18)
    # 实例化损失函数对象
    my_null_loss = nn.NLLLoss()
    # 实例化优化器对象
    my_optim = optim.Adam(my_gru.parameters(), lr=my_lr)

    # 定义打印日本函数
    start_time = time.time()
    total_iter_num = 0  # 已经训练样本的总数
    total_loss = 0  # 已经训练的损失
    total_loss_list = []  # 每隔n个样本，保存平均损失
    total_acc_num = 0  # 预测正确的样本个数
    total_acc_list = []  # 每隔n个样本，保存平均准确率
    # 开始训练
    for epoch_idx in range(epochs):
        # 实例化dataloader
        my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
        for i, (x, y) in enumerate(tqdm(my_dataloader)):
            # print(f"x.shape-->{x.shape}")
            # print(f"y-->{y}")
            output, hn = my_gru(input=x[0], hidden=my_gru.initthidden())
            # print(f'output.shape-->{output.shape}')  # [1, 18]
            # 计算损失
            my_loss = my_null_loss(output, y)
            # print(f'my_loss-->{my_loss}')
            # 梯度清零,因为是用优化器对梯度更新的，所以也要用优化器对梯度清零
            my_optim.zero_grad()
            # 反向传播
            my_loss.backward()
            # 梯度更新
            my_optim.step()

            # 统计下已经训练样本的总个数
            total_iter_num = total_iter_num + 1
            # 统计一下已经训练样本的总损失
            total_loss = total_loss + my_loss.item()
            # 统计已经训练的样本中预测正确的个数
            # a = torch.argmax(output)
            # print(f'a-->{a}')
            i_predict_num = 1 if torch.argmax(output).item() == y.item() else 0
            total_acc_num += i_predict_num
            # print(f"torch.argmax(output).item()-->{torch.argmax(output).item()}")
            # print(f"y-->{y}")
            # print(f'i_predict_num-->{i_predict_num}')

            # 每隔100次训练，保存一下平均损失和准确率
            if total_iter_num % 100 == 0:
                avg_loss = total_loss / total_iter_num
                total_loss_list.append(avg_loss)

                avg_acc = total_acc_num / total_iter_num
                total_acc_list.append(avg_acc)
            # 每隔2000次训练，打印一次日志
            if total_iter_num % 2000 == 0:
                temp_loss = total_loss / total_iter_num
                temp_acc = total_acc_num / total_iter_num
                temp_time = time.time() - start_time
                print('轮次：%d，损失：%.6f,时间：%d,准确率:%.3f' % (epoch_idx + 1, temp_loss, temp_time, temp_acc))
            pred = torch.argmax(output).item()

            # print(
            #     f"预测:{categorys[pred]} "
            #     f"真实:{categorys[y.item()]}"
            # )

        # 每个轮次保存模型
        torch.save(my_gru.state_dict(), './save_model/my_gru_model_%d.bin' % (epoch_idx + 1))
    # 计算总时间
    total_time = int(time.time() - start_time)
    # print(f'total_time-->{total_time}')
    # 将结果保存到文件中
    save_data = {
        "loss": total_loss_list,
        "time": total_time,
        "acc": total_acc_list
    }

    with open('./save_results/ai_gru_train_result.json', 'w', encoding='utf-8') as fw:
        json.dump(save_data, fw, ensure_ascii=False, indent=4)
    return total_loss_list, total_time, total_acc_list


def read_json(data_path):
    with open(data_path, 'r') as fr:
        results = json.loads(fr.read())
    avg_loss = results["loss"]
    all_time = results["time"]
    avg_acc = results["acc"]
    return avg_loss, all_time, avg_acc


def dm_show_results():
    rnn_avg_loss, rnn_all_time, rnn_avg_acc = read_json("./save_results/ai_rnn_train_result.json")
    lstm_avg_loss, lstm_all_time, lstm_avg_acc = read_json("./save_results/ai_lstm_train_result.json")
    gru_avg_loss, gru_all_time, gru_avg_acc = read_json("./save_results/ai_gru_train_result.json")
    print(rnn_all_time)
    print(lstm_all_time)
    print(gru_all_time)
    print(rnn_avg_acc)
    # 对于不同模型的损失
    plt.figure(0)
    plt.plot(rnn_avg_loss, label="RNN")
    plt.plot(lstm_avg_loss, label="LSTM", color='red')
    plt.plot(gru_avg_loss, label="GRU", color='orange')
    plt.legend(loc='upper left')
    plt.savefig('./img/loss.png')
    plt.show()

    # 对比不同模型的耗时
    plt.figure(1)
    x_data = ["RNN", "LSTM", "GRU"]
    y_data = [rnn_all_time, lstm_all_time, gru_all_time]
    plt.bar(range(len(x_data)), y_data, tick_label=x_data)
    plt.savefig('./img/time.png')
    plt.show()

    # 对于不同模型的准确率
    plt.figure(2)
    plt.plot(rnn_avg_acc, label="RNN")
    plt.plot(lstm_avg_acc, label="LSTM", color='red')
    plt.plot(gru_avg_acc, label="GRU", color='orange')
    plt.legend(loc='upper left')
    plt.savefig('./img/acc.png')
    plt.show()


# 构造将字符串（人名）转换为张量的函数
def line2tensor(x):
    tensor_x = torch.zeros(len(x), n_letters)
    for li, letter in enumerate(x):
        tensor_x[li][letter.find(letter)] = 1
    return tensor_x


rnn_model_path = "./save_model/my_rnn_model_1.bin"
lstm_model_path = "./save_model/my_lstm_model_1.bin"
gru_model_path = "./save_model/my_rnn_model_1.bin"


def rnn_predict(x):
    tensor_x = line2tensor(x)
    my_rnn = My_RNN(input_size=57, hidden_size=128, output_size=18)
    my_rnn.load_state_dict(torch.load(rnn_model_path))
    with torch.no_grad():
        # 将数据送入模型
        output, hn = my_rnn(tensor_x, my_rnn.initthidden())
        # print(f"output-->{output}")
        values, indexes = torch.topk(output, k=3)
        print(f"values-->{values}")
        print(f"indexes-->{indexes}")
        for i in range(3):
            value = values[0][i]
            index = indexes[0][1]
            category = categorys[index]
            print(f"当前预测的值是：{value}    预测结果是：{category}")


def lstm_predict(x):
    tensor_x = line2tensor(x)
    my_lstm = My_LSTM(input_size=57, hidden_size=128, output_size=18)
    my_lstm.load_state_dict(torch.load(lstm_model_path))
    with torch.no_grad():
        # 将数据送入模型
        hidden, c = my_lstm.initthidden()
        output, hn, cn = my_lstm(tensor_x, hidden, c)
        # print(f"output-->{output}")
        values, indexes = torch.topk(output, k=3)
        print(f"values-->{values}")
        print(f"indexes-->{indexes}")
        for i in range(3):
            value = values[0][i]
            index = indexes[0][1]
            category = categorys[index]
            print(f"当前预测的值是：{value}    预测结果是：{category}")


def gru_predict(x):
    tensor_x = line2tensor(x)
    my_gru = My_RNN(input_size=57, hidden_size=128, output_size=18)
    my_gru.load_state_dict(torch.load(gru_model_path))
    with torch.no_grad():
        # 将数据送入模型
        output, hn = my_gru(tensor_x, my_gru.initthidden())
        # print(f"output-->{output}")
        values, indexes = torch.topk(output, k=3)
        print(f"values-->{values}")
        print(f"indexes-->{indexes}")
        for i in range(3):
            value = values[0][i]
            index = indexes[0][1]
            category = categorys[index]
            print(f"当前预测的值是：{value}    预测结果是：{category}")


if __name__ == '__main__':
    # test_dataset()
    # print(my_list_x, my_list_y)
    # get_dataloader()
    # test_model_RNN()
    # test_model_LSTM()
    # test_model_GRU()
    # train_rnn()
    # train_lstm()
    # train_gru()
    # dm_show_results()
    # result = line2tensor("bai")
    # print(result)
    name = "zhang"
    rnn_predict(name)
    print("*" * 40)
    lstm_predict(name)
    print("*" * 40)
    gru_predict(name)
