# Adaptive fault diagnosis based on counter transfer learning
**基于对抗迁移学习的故障自适应诊断**

**1.简介**

提出GDANN模型，解决轴承故障诊断的带载训练、空载测试情况下准确率较低的问题。

**2.硬件**

CPU：i7-8550U，显卡：MX150

**3.框架**

Keras，Sklearn

**4.依赖**

tensorflow 2.0;keras;numpy;scipy;os;sklearn;matplotlib

**5.说明**

model_GDANN.py 构建GDANN网络的模型

model_DANN.py 构建DANN网络的模型

model_train_test.py 利用前两个模型训练和测试的文件

model_checkpoint.py 将模型和训练测试代码放在一起，实现断点续训功能

*checkpoint文件夹：保存模型参数，断点续训使用；
data文件夹：CWRU数据集；
log文件夹：保存日志文件，可用tensorboard打开查看acc、loss曲线*
