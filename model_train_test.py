from model_GDANN import DANN_Model
import preprocess
import numpy as np
import datetime

############################################ 训练、测试部分 ####################################
# 训练参数
batch_size = 32
EPOCHS = 602
num_classes = 10
length = 4096
BatchNorm = True # 是否批量归一化
number = 1000 # 每类样本的数量
normal = True # 是否标准化
rate = [0.7,0.2,0.1] # 测试集验证集划分比例

path0 = r'data\3HP'  #源域
path1 = r'data\0HP'  #目标域


run_name='changshi'

########################### 源域数据集加载 ####################
x_train0, y_train0, x_valid0, y_valid0, x_test0, y_test0 = preprocess.prepro(d_path=path0,length=length,
                                                                  number=number,
                                                                  normal=normal,
                                                                  rate=rate,
                                                                  enc=True, enc_step=28) #最后俩个：是否数据增强，数据增强的顺延间隔

#插入新维度，方便卷积网络输入
x_train0, x_valid0, x_test0 = x_train0[:,:,np.newaxis], x_valid0[:,:,np.newaxis], x_test0[:,:,np.newaxis]
# 输入数据的维度
input_shape =x_train0.shape[1:]

########################### 目标域数据集加载1 ####################
x_train1, y_train1, x_valid1, y_valid1, x_test1, y_test1 = preprocess.prepro(d_path=path1,length=length,
                                                                  number=number,
                                                                  normal=normal,
                                                                  rate=rate,
                                                                  enc=True, enc_step=28) #最后俩个：是否数据增强，数据增强的顺延间隔

#插入新维度，方便卷积网络输入
x_train1, x_valid1, x_test1 = x_train1[:,:,np.newaxis], x_valid1[:,:,np.newaxis], x_test1[:,:,np.newaxis]

#将数据集转换为模型需要的tf.dataset格式
(source_train_dataset, source_test_dataset)=preprocess.pre_batch(x_train0, y_train0, x_test0, y_test0, batch_size)
(target_train_dataset, target_test_dataset)=preprocess.pre_batch(x_train1, y_train1, x_test1, y_test1, batch_size)

#学习率
fe_lr = 0.005  #特征提取器
lp_lr = 0.005  #标签分类器
dc_lr = 0.0005  #域判别器

lr = (lp_lr, dc_lr, fe_lr)
model = DANN_Model(input_shape=input_shape, model_type='no_load', run_name=run_name, lr=lr)


for epoch in range(EPOCHS):

    print(datetime.datetime.now())

    for (source_samples, class_labels), (target_samples, _) in zip(source_train_dataset, target_train_dataset):
        model.train(source_samples, class_labels, target_samples)

    latent_source = []
    latent_target = []
    for (test_samples, test_labels), (target_test_samples, target_test_labels) in zip(source_test_dataset,
                                                                                    target_test_dataset):
        model.test_source(test_samples, test_labels, target_test_samples)
        model.test_target(target_test_samples, target_test_labels)

        if len(latent_source) == 0:
            latent_source = model.return_latent_variables(test_samples)
        else:
            latent_source = np.concatenate([latent_source, model.return_latent_variables(test_samples)])

        if len(latent_target) == 0:
            latent_target = model.return_latent_variables(target_test_samples)
        else:
            latent_target = np.concatenate([latent_target, model.return_latent_variables(target_test_samples)])

    print('Epoch: {}'.format(epoch + 1))
    print(model.log())

# checkpoint = tf.train.Checkpoint(myModel=model)             # 实例化Checkpoint，指定恢复对象为model
# print(tf.train.latest_checkpoint('./checkpoint/original-3_to_0HP-lp_conv-1'))
# checkpoint.restore(tf.train.latest_checkpoint('./checkpoint/original-3_to_0HP-lp_conv-1'))    # 从文件恢复模型参数
# for (test_samples, test_labels), (target_test_samples, target_test_labels) in zip(source_test_dataset,
#                                                                                 target_test_dataset):
#     model.test_source(test_samples, test_labels, target_test_samples)
#     model.test_target(target_test_samples, target_test_labels)
# print(model.log())

# index = [0, len(latent_source), len(latent_source) + len(latent_target)]
# latent_variables = np.concatenate([latent_source, latent_target])
#
# pca_embedding = PCA(n_components=2).fit_transform(latent_variables)
#
# plt.figure()
# plt.title('Epoch #{}'.format(epoch + 1))
# for i in range(len(index) - 1):
#     plt.plot(pca_embedding[index[i]:index[i + 1], 0], pca_embedding[index[i]:index[i + 1], 1], '.', alpha=0.5)
# plt.legend(['source', 'target'])
# plt.show()


