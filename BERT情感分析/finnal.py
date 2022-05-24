import torch
import time
import numpy as np
import pandas as pd
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


class Config:
    def __init__(self):
        """
        参数设置
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch设备选择CPU/GPU
        self.csv_path = '初步处理的数据_八达岭长城.csv'  # 训练数据路径
        self.D_in, self.H, self.D_out = 768, 100, 2  # 输入维度,隐藏层维度,输出维度
        self.batch_size = 32  # Batch size for training.
        self.epochs = 4  # Number of training epochs.
        self.lr = 5e-5  # Learning rate for AdamW.
        self.eps = 1e-8  # AdamW epsilon.
        self.MAX_LEN = 256  # Maximum length of the input.


class BertDataProcessing(Config):
    def __init__(self):
        super(BertDataProcessing, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')  # 加载预训练模型
        self.data = pd.read_csv(self.csv_path)  # 读取训练数据

    def split_data(self):
        """
        将数据集分为训练集和测试集
        :return:
        """
        X = self.data['评论'].values  # comment
        y = self.data['label'].values  # label自己给的0 1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)  # 分割训练集和测试集
        return X_train, X_test, y_train, y_test

    def data_preprocessing(self, data):
        """
        数据预处理
        :param data: 需要预处理的数据
        :return: 返回预处理后的数据
        """
        # 对训练集和测试集进行预处理
        # 空列表来储存信息
        input_ids = []  # 输入ids
        attention_masks = []  # attention_mask

        # 每个句子循环一次
        for sent in data:
            encoded_sent = self.tokenizer.encode_plus(
                text=sent,  # 预处理语句
                add_special_tokens=True,  # 加 [CLS] 和 [SEP]
                max_length=self.MAX_LEN,  # 截断或者填充的最大长度
                padding='max_length',  # 填充为最大长度，这里的padding在之间可以直接用pad_to_max但是版本更新之后弃用了，老版本什么都没有，可以尝试用extend方法
                truncation=True,  # 文本太长，需要截断操作
                return_attention_mask=True  # 返回 attention mask
            )

            # 把输出加到列表里面
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

        # 把list转换为tensor
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return input_ids, attention_masks

    def data_loader(self):
        """Create PyTorch DataLoader

        我们将使用torch DataLoader类为数据集创建一个迭代器。这将有助于在训练期间节省内存并提高训练速度。

        """
        X_train, X_test, y_train, y_test = self.split_data()
        # 在train，test上运行 preprocessing_for_bert 转化为指定输入格式
        train_inputs, train_masks = self.data_preprocessing(X_train)
        test_inputs, test_masks = self.data_preprocessing(X_test)

        # 转化为tensor类型

        train_labels = torch.tensor(y_train, dtype=torch.long)
        test_labels = torch.tensor(y_test, dtype=torch.long)

        # 给训练集创建 DataLoader
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)
        # print(train_dataloader)

        # 给验证集创建 DataLoader
        test_data = TensorDataset(test_inputs, test_masks, test_labels)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.batch_size)
        return train_dataloader, test_dataloader


# 自己定义的Bert分类器的类，微调Bert模型
class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        """
        freeze_bert (bool): 设置是否进行微调，0就是不，1就是调
        """
        super(BertClassifier, self).__init__()
        # 输入维度(hidden size of Bert)默认768，分类器隐藏维度，输出维度(label)
        D_in, H, D_out = Config().D_in, Config().H, Config().D_out

        # 实体化Bert模型
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # 实体化一个单层前馈分类器，说白了就是最后要输出的时候搞个全连接层
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),  # 全连接
            nn.ReLU(),  # 激活函数
            nn.Dropout(0.5),
            nn.Linear(H, D_out)  # 全连接
        )
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_gard = False

    def forward(self, input_ids, attention_mask):
        # 开始搭建整个网络了
        # 输入
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        # 为分类任务提取标记[CLS]的最后隐藏状态，因为要连接传到全连接层去
        last_hidden_state_cls = outputs[0][:, 0, :]
        # 全连接，计算，输出label
        logits = self.classifier(last_hidden_state_cls)

        return logits


class Train(Config):
    def __init__(self):
        super(Train, self).__init__()
        # 初始化我们的Bert分类器
        self.model = BertClassifier()
        self.train_dataloader, self.test_dataloader = BertDataProcessing().data_loader()  # 获取训练集和测试集
        # 用GPU运算
        self.model.to(self.device)
        # 创建优化器
        self.optimizer = AdamW(self.model.parameters(),
                               lr=self.lr,  # 默认学习率
                               eps=self.eps  # 默认精度
                               )
        # 训练的总步数
        self.total_steps = len(self.train_dataloader) * self.epochs
        # 学习率预热
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,  # Default value
                                                         num_training_steps=self.total_steps)
        self.loss_fn = nn.CrossEntropyLoss()  # 交叉熵

    def evaluate(self):
        """
        在每个epoch后验证集上评估model性能
        """
        # model放入评估模式
        model = self.model
        model.eval()

        # 准确率和误差
        test_accuracy = []
        test_loss = []

        # 验证集上的每个batch
        for batch in self.test_dataloader:
            # 放到GPU上
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)

            # 计算结果，不计算梯度
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)  # 放到model里面去跑，返回验证集的ouput就是一行三列的
                # label向量可能性，这个时候还没有归一化所以还不能说是可能性，反正归一化之后最大的就是了

            # 计算误差
            loss = self.loss_fn(logits, b_labels.long())
            test_loss.append(loss.item())

            # get预测结果，这里就是求每行最大的索引咯，然后用flatten打平成一维
            preds = torch.argmax(logits, dim=1).flatten()  # 返回一行中最大值的序号

            # 计算准确率，这个就是俩比较，返回相同的个数, .cpu().numpy()就是把tensor从显卡上取出来然后转化为numpy类型的举证好用方法
            # 最后mean因为直接bool形了，也就是如果预测和label一样那就返回1，正好是正确的个数，求平均就是准确率了
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            test_accuracy.append(accuracy)

        # 计算整体的平均正确率和loss
        val_loss = np.mean(test_loss)
        val_accuracy = np.mean(test_accuracy)

        return val_loss, val_accuracy

    def train(self, evaluation=False):
        # 开始训练循环
        for epoch_i in range(self.epochs):
            # =======================================
            #               Training
            # =======================================
            # 表头
            print(
                f"{'Epoch':^7} | {'每40个Batch':^9} | {'Batch总数':^10}|{'训练集 Loss':^12} | {'测试集 Loss':^10} | {'测试集准确率':^9} | {'时间':^9}")
            print("-" * 80)

            # 测量每个epoch经过的时间
            t0_epoch, t0_batch = time.time(), time.time()

            # 在每个epoch开始时重置跟踪变量
            total_loss, batch_loss, batch_counts = 0, 0, 0

            # 把model放到训练模式
            model = self.model
            model.train()

            # 分batch训练
            for step, batch in enumerate(self.train_dataloader):
                batch_counts += 1
                # 把batch加载到GPU
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)
                # print(b_labels.shape)
                # 归零导数
                model.zero_grad()
                # 真正的训练
                logits = model(b_input_ids, b_attn_mask)
                # print(logits.shape)
                # 计算loss并且累加

                loss = self.loss_fn(logits, b_labels)

                batch_loss += loss.item()
                total_loss += loss.item()
                # 反向传播
                loss.backward()
                # 归一化，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # 更新参数和学习率
                self.optimizer.step()
                self.scheduler.step()
                # 清空不需要的变量，防止内存泄漏
                del loss
                torch.cuda.empty_cache()

                # Print每40个batch的loss和time
                if (step % 40 == 0 and step != 0) or (step == len(self.train_dataloader) - 1):
                    # 计算40个batch的时间
                    time_elapsed = time.time() - t0_batch

                    # Print训练结果
                    print(
                        f"{epoch_i + 1:^7} | {step:^10} | {len(self.train_dataloader):^10}|{batch_loss / batch_counts:^14.6f} | {'-':^12} | {'-':^13} | {time_elapsed:^9.2f}")

                    # 重置batch参数
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

            # 计算平均loss 这个是训练集的loss
            avg_train_loss = total_loss / len(self.train_dataloader)

            print("-" * 80)
            # =======================================
            #               Evaluation
            # =======================================
            if evaluation:  # 这个evalution是我们自己给的，用来判断是否需要我们汇总评估
                # 每个epoch之后评估一下性能
                # 在我们的验证集/测试集上.
                test_loss, test_accuracy = self.evaluate()
                # Print 整个训练集的耗时
                time_elapsed = time.time() - t0_epoch

                print(
                    f"{epoch_i + 1:^7} | {'-':^10}| {'-':^10} | {avg_train_loss:^14.6f} | {test_loss:^12.6f} | {test_accuracy:^12.2f}% | {time_elapsed:^9.2f}")
                print("-" * 80)
            print("\n")


if __name__ == '__main__':
    Train().train(evaluation=True)
