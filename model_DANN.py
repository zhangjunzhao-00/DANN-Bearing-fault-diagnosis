import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
import tensorflow.keras as keras


MAX_STEP = 10000

log_format = 'L1 Test: {:.4f}, Acc1 Test: {:.2f}\n' + \
             'L2 Test: {:.4f}, Acc2 Test: {:.2f}\n' + \
             'L3 Test: {:.4f}, Acc3 Test: {:.2f}\n'

log_format_source_only = 'L1 Test: {:.4f}, Acc1 Test: {:.2f}\n' + \
                         'L1 Target_1: {:.4f}, Acc3 Target_1: {:.2f}\n' + \
                         'L1 Target_2: {:.4f}, Acc3 Target_2: {:.2f}\n'


@tf.custom_gradient
def GradientReversalOperator(x):  # 实现梯度反转
    def grad(dy):
        return -1 * dy

    return x, grad


class GradientReversalLayer(tf.keras.layers.Layer):  # 构建梯度反转层
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def call(self, inputs):
        return GradientReversalOperator(inputs)


class DANN_Model(keras.Model):
    def __init__(self, input_shape, model_type, run_name, lr, source_only=False, category=(None, None)):
        super(DANN_Model, self).__init__()

        if (model_type == 'no_load'):
            self.feature_extractor = Sequential([
                # 第一层
                Conv1D(filters=32, kernel_size=20, strides=8, padding='same', kernel_regularizer=l2(1e-4),
                       input_shape=input_shape),
                BatchNormalization(),
                Activation('relu'),
                MaxPooling1D(pool_size=4, strides=4, padding='valid'),
                # 第二层
                Conv1D(filters=64, kernel_size=5, strides=2, padding='same', kernel_regularizer=l2(1e-4)),
                BatchNormalization(),
                Activation('relu'),
                MaxPooling1D(pool_size=2, strides=2, padding='valid'),
            ])

            self.label_predictor = Sequential([
                Dropout(0.1),
                # 第三层
                Conv1D(filters=64, kernel_size=5, strides=2, padding='same', kernel_regularizer=l2(1e-4)),
                BatchNormalization(),
                Activation('relu'),
                MaxPooling1D(pool_size=2, strides=2, padding='valid'),
                Flatten(),
                Dropout(0.1),
                Dense(units=100, kernel_regularizer=l2(1e-4)),
                # BatchNormalization(),
                Activation('relu'),
                Dropout(0.1),
                # Dense(100),
                # BatchNormalization(),
                # Activation('relu'),
                # Dropout(0.5),
                Dense(units=10, activation='softmax', kernel_regularizer=l2(1e-4))
            ])

            self.domain_classifier = Sequential([
                Flatten(),
                GradientReversalLayer(),
                Dense(units=100),
                # BatchNormalization(),
                Activation('relu'),
                # Dropout(0.5),
                Dense(units=2, activation='sigmoid'),
                # Activation('sigmoid')
            ])

        # elif (model_type == 'on_load'):
        #
        # 	self.feature_extractor = Sequential([
        # 		Conv2D(filters=64, kernel_size=5, strides=1, kernel_regularizer=l2(0.001), padding='same', input_shape=input_shape),
        # 		BatchNormalization(),
        # 		Activation('relu'),
        # 		MaxPooling2D(pool_size=3, strides=2),
        # 		Conv2D(filters=64, kernel_size=5, padding='same', strides=1, kernel_regularizer=l2(0.001)),
        # 		BatchNormalization(),
        # 		Activation('relu'),
        # 		MaxPooling2D(pool_size=3, strides=2),
        # 		Conv2D(filters=128, kernel_size=5, padding='same', strides=1, kernel_regularizer=l2(0.001)),
        # 		BatchNormalization(),
        # 		Activation('relu'),
        # 		Flatten()
        # 	])
        #
        # 	self.label_predictor = Sequential([
        # 		Dense(3072, kernel_regularizer=l2(0.001)),
        # 		BatchNormalization(),
        # 		Activation('relu'),
        # 		Dropout(0.5),
        # 		Dense(2048, kernel_regularizer=l2(0.001)),
        # 		BatchNormalization(),
        # 		Activation('relu'),
        # 		Dropout(0.5),
        # 		Dense(10, activation='softmax')
        # 	])
        #
        # 	self.domain_classifier = Sequential([
        # 		GradientReversalLayer(),
        # 		Dense(1024, kernel_regularizer=l2(0.001)),
        # 		BatchNormalization(),
        # 		Activation('relu'),
        # 		Dropout(0.5),
        # 		Dense(1024, kernel_regularizer=l2(0.001)),
        # 		BatchNormalization(),
        # 		Activation('relu'),
        # 		Dropout(0.5),
        # 		Dense(2, kernel_regularizer=l2(0.001)),
        # 		Activation('softmax')
        # 	])

        self.predict_label = Sequential([
            self.feature_extractor,
            self.label_predictor
        ])

        self.classify_domain = Sequential([
            self.feature_extractor,
            self.domain_classifier
        ])

        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.dcloss = tf.keras.losses.CategoricalCrossentropy()

        self.lp_lr = lr[0]
        self.dc_lr = lr[1]
        self.fe_lr = lr[2]

        self.lp_optimizer = tf.keras.optimizers.Adam(learning_rate=lr[0])
        self.dc_optimizer = tf.keras.optimizers.Adam(learning_rate=lr[1])
        self.fe_optimizer = tf.keras.optimizers.Adam(learning_rate=lr[2])  # fe_lr=0.0005

        # self.lp_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lp_lr, decay=0.00005)
        # self.dc_optimizer = tf.keras.optimizers.Adam(learning_rate=self.dc_lr, decay=0.0002)
        # self.fe_optimizer = tf.keras.optimizers.Adam(learning_rate=self.fe_lr, decay=0.0002) #fe_lr=0.0005

        self.train_lp_loss = tf.keras.metrics.Mean()
        self.train_dc_loss = tf.keras.metrics.Mean()

        self.train_lp_accuracy = tf.keras.metrics.Accuracy()
        self.train_dc_accuracy = tf.keras.metrics.Accuracy()

        self.test_lp_loss = tf.keras.metrics.Mean()
        self.test_dc_loss = tf.keras.metrics.Mean()
        self.test_target_lp_loss = tf.keras.metrics.Mean()

        self.test_lp_accuracy = tf.keras.metrics.Accuracy()
        self.test_dc_accuracy = tf.keras.metrics.Accuracy()
        self.test_target_lp_accuracy = tf.keras.metrics.Accuracy()
        self.train_target_lp_accuracy = tf.keras.metrics.Accuracy()

        if source_only:
            self.target_accuracy_1 = tf.keras.metrics.Accuracy()
            self.target_accuracy_2 = tf.keras.metrics.Accuracy()
            self.target_loss_1 = tf.keras.metrics.Mean()
            self.target_loss_2 = tf.keras.metrics.Mean()

        self.create_logger(run_name, source_only, category)

    @tf.function
    def train(self, x_class, y_class, x_domain):

        domain_labels = np.concatenate([np.zeros(len(x_class)), np.ones(len(x_domain))])
        domain_labels = tf.one_hot(domain_labels, 2)  # 将域标签从64行一维张量转换为(64,2)的独热码形式的二维张量

        x_both = tf.concat([x_class, x_domain], axis=0)

        with tf.GradientTape() as tape:
            y_class_pred = self.predict_label(x_class, training=True)
            lp_loss = self.loss(y_class, y_class_pred)
        lp_grad = tape.gradient(lp_loss, self.predict_label.trainable_variables)

        with tf.GradientTape(persistent=True) as tape:
            y_domain_pred = self.classify_domain(x_both, training=True)
            dc_loss = self.dcloss(domain_labels, y_domain_pred)
        fe_grad = tape.gradient(dc_loss, self.feature_extractor.trainable_variables)
        dc_grad = tape.gradient(dc_loss, self.domain_classifier.trainable_variables)
        del tape

        self.lp_optimizer.apply_gradients(zip(lp_grad, self.predict_label.trainable_variables))
        # self.dc_optimizer.apply_gradients(zip(dc_grad, self.classify_domain.trainable_variables))
        self.dc_optimizer.apply_gradients(zip(dc_grad, self.domain_classifier.trainable_variables))

        self.fe_optimizer.apply_gradients(zip(fe_grad, self.feature_extractor.trainable_variables))

        self.train_lp_loss(lp_loss)
        self.train_lp_accuracy(y_class, y_class_pred)

        self.train_dc_loss(dc_loss)
        self.train_dc_accuracy(domain_labels, y_domain_pred)

        return

    @tf.function
    def train_source_only(self, x_class, y_class):

        with tf.GradientTape(persistent=True) as tape:
            y_class_pred = self.predict_label(x_class, training=True)
            lp_loss = self.loss(y_class, y_class_pred)
        lp_grad = tape.gradient(lp_loss, self.predict_label.trainable_variables)
        fe_grad = tape.gradient(lp_loss, self.feature_extractor.trainable_variables)
        del tape

        self.lp_optimizer.apply_gradients(zip(lp_grad, self.predict_label.trainable_variables))
        self.fe_optimizer.apply_gradients(zip(fe_grad, self.feature_extractor.trainable_variables))

        self.train_lp_loss(lp_loss)
        self.train_lp_accuracy(y_class, y_class_pred)

        return

    @tf.function
    def test(self, x_class, y_class, x_domain, y_domain):

        domain_labels = np.concatenate([np.zeros(len(x_class)), np.ones(len(x_domain))])
        domain_labels = tf.one_hot(domain_labels, 2)  # 将域标签从64行一维张量转换为(64,2)的独热码形式的二维张量

        x_both = tf.concat([x_class, x_domain], axis=0)

        with tf.GradientTape() as tape:
            y_class_pred = self.predict_label(x_class, training=False)
            y_domain_pred = self.classify_domain(x_both, training=False)
            y_target_class_pred = self.predict_label(x_domain, training=False)

            lp_loss = self.loss(y_class, y_class_pred)
            dc_loss = self.dcloss(domain_labels, y_domain_pred)
            target_lp_loss = self.loss(y_domain, y_target_class_pred)

        self.test_lp_loss(lp_loss)
        self.test_lp_accuracy(y_class, y_class_pred)

        self.test_dc_loss(dc_loss)
        self.test_dc_accuracy(domain_labels, y_domain_pred)

        self.test_target_lp_loss(target_lp_loss)
        self.test_target_lp_accuracy(y_domain, y_target_class_pred)

        return

    @tf.function
    def test_source(self, x_class, y_class, x_domain):

        domain_labels = np.concatenate([np.zeros(len(x_class)), np.ones(len(x_domain))])
        domain_labels = tf.one_hot(domain_labels, 2)  # 将域标签从64行一维张量转换为(64,2)的独热码形式的二维张量

        x_both = tf.concat([x_class, x_domain], axis=0)

        with tf.GradientTape() as tape:
            y_class_pred = self.predict_label(x_class, training=False)
            y_domain_pred = self.classify_domain(x_both, training=False)
            lp_loss = self.loss(y_class, y_class_pred)
            dc_loss = self.dcloss(domain_labels, y_domain_pred)

        self.test_lp_loss(lp_loss)
        self.test_lp_accuracy(y_class, y_class_pred)

        self.test_dc_loss(dc_loss)
        self.test_dc_accuracy(domain_labels, y_domain_pred)

        return

    @tf.function
    def test_target(self, x_domain, y_domain):

        with tf.GradientTape() as tape:
            y_target_class_pred = self.predict_label(x_domain, training=False)
            target_lp_loss = self.loss(y_domain, y_target_class_pred)

        self.test_target_lp_loss(target_lp_loss)
        self.test_target_lp_accuracy(y_domain, y_target_class_pred)

        return

    @tf.function
    def test_source_only(self, x_domain, y_domain, mode):

        with tf.GradientTape() as tape:
            y_target_class_pred = self.predict_label(x_domain, training=False)
            target_lp_loss = self.loss(y_domain, y_target_class_pred)

        if mode == 0:
            self.test_lp_loss(target_lp_loss)
            self.test_lp_accuracy(y_domain, y_target_class_pred)

        if mode == 1:
            self.target_loss_1(target_lp_loss)
            self.target_accuracy_1(y_domain, y_target_class_pred)

        elif mode == 2:
            self.target_loss_2(target_lp_loss)
            self.target_accuracy_2(y_domain, y_target_class_pred)

        return

    def return_latent_variables(self, x_domain):

        latent_variable = self.feature_extractor(x_domain, training=False)

        return latent_variable

    def log(self):
        message = log_format.format(
            self.test_lp_loss.result(),
            self.test_lp_accuracy.result() * 100,
            self.test_dc_loss.result(),
            self.test_dc_accuracy.result() * 100,
            self.test_target_lp_loss.result(),
            self.test_target_lp_accuracy.result() * 100)

        with self.train_writer.as_default():
            tf.summary.scalar('label_prediction_loss', self.train_lp_loss.result(), step=self.lp_optimizer.iterations)
            tf.summary.scalar('label_prediction_accuracy', self.train_lp_accuracy.result(),
                              step=self.lp_optimizer.iterations)
            tf.summary.scalar('domain_classification_loss', self.train_dc_loss.result(),
                              step=self.lp_optimizer.iterations)
            tf.summary.scalar('domain_classification_accuracy', self.train_dc_accuracy.result(),
                              step=self.lp_optimizer.iterations)

        with self.test_writer.as_default():
            tf.summary.scalar('label_prediction_loss', self.test_lp_loss.result(), step=self.lp_optimizer.iterations)
            tf.summary.scalar('label_prediction_accuracy', self.test_lp_accuracy.result(),
                              step=self.lp_optimizer.iterations)
            tf.summary.scalar('domain_classification_loss', self.test_dc_loss.result(),
                              step=self.lp_optimizer.iterations)
            tf.summary.scalar('domain_classification_accuracy', self.test_dc_accuracy.result(),
                              step=self.lp_optimizer.iterations)

        with self.target_writer.as_default():
            tf.summary.scalar('label_prediction_loss', self.test_target_lp_loss.result(),
                              step=self.lp_optimizer.iterations)
            tf.summary.scalar('label_prediction_accuracy', self.test_target_lp_accuracy.result(),
                              step=self.lp_optimizer.iterations)

        self.reset_metrics('train')
        self.reset_metrics('test')

        return message

    def log_source_only(self):
        message = log_format.format(
            self.test_lp_loss.result(),
            self.test_lp_accuracy.result() * 100,
            self.target_loss_1.result(),
            self.target_accuracy_1.result() * 100,
            self.target_loss_2.result(),
            self.target_accuracy_2.result() * 100)

        with self.train_writer.as_default():
            tf.summary.scalar('label_prediction_loss', self.train_lp_loss.result(), step=self.lp_optimizer.iterations)
            tf.summary.scalar('label_prediction_accuracy', self.train_lp_accuracy.result(),
                              step=self.lp_optimizer.iterations)

        with self.test_writer.as_default():
            tf.summary.scalar('label_prediction_loss', self.test_lp_loss.result(), step=self.lp_optimizer.iterations)
            tf.summary.scalar('label_prediction_accuracy', self.test_lp_accuracy.result(),
                              step=self.lp_optimizer.iterations)

        with self.target_writer_1.as_default():
            tf.summary.scalar('label_prediction_loss', self.target_loss_1.result(), step=self.lp_optimizer.iterations)
            tf.summary.scalar('label_prediction_accuracy', self.target_accuracy_1.result(),
                              step=self.lp_optimizer.iterations)

        with self.target_writer_2.as_default():
            tf.summary.scalar('label_prediction_loss', self.target_loss_2.result(), step=self.lp_optimizer.iterations)
            tf.summary.scalar('label_prediction_accuracy', self.target_accuracy_2.result(),
                              step=self.lp_optimizer.iterations)

        self.reset_metrics('source_only')

        return message

    def create_logger(self, run_name, source_only, category):
        if os.path.isdir("./log/{}".format(run_name)):
            for i in range(99):
                if not os.path.isdir("./log/{}_{}".format(run_name, i)):
                    run_name = '{}_{}'.format(run_name, i)
                    break

        run_dir = "./log/{}".format(run_name)
        train_dir = "./log/{}/train".format(run_name)
        test_dir = "./log/{}/test".format(run_name)

        os.mkdir(run_dir)
        os.mkdir(train_dir)
        os.mkdir(test_dir)

        self.train_writer = tf.summary.create_file_writer(train_dir)
        self.test_writer = tf.summary.create_file_writer(test_dir)

        if source_only:
            target_dir_1 = "./log/{}/target0_{}".format(run_name, category[0])
            target_dir_2 = "./log/{}/target1_{}".format(run_name, category[1])
            os.mkdir(target_dir_1)
            os.mkdir(target_dir_2)
            self.target_writer_1 = tf.summary.create_file_writer(target_dir_1)
            self.target_writer_2 = tf.summary.create_file_writer(target_dir_2)

        else:
            target_dir = "./log/{}/target".format(run_name)
            os.mkdir(target_dir)
            self.target_writer = tf.summary.create_file_writer(target_dir)

        print("Log folder created as {}".format(run_dir))

        return

    def reset_metrics(self, target):

        if target == 'train':
            self.train_lp_loss.reset_states()
            self.train_lp_accuracy.reset_states()
            self.train_dc_loss.reset_states()
            self.train_dc_accuracy.reset_states()

        if target == 'test':
            self.test_lp_loss.reset_states()
            self.test_lp_accuracy.reset_states()
            self.test_dc_loss.reset_states()
            self.test_dc_accuracy.reset_states()
            self.test_target_lp_loss.reset_states()
            self.test_target_lp_accuracy.reset_states()

        elif target == 'source_only':
            self.train_lp_loss.reset_states()
            self.train_lp_accuracy.reset_states()
            self.test_lp_loss.reset_states()
            self.test_lp_accuracy.reset_states()
            self.target_loss_1.reset_states()
            self.target_accuracy_1.reset_states()
            self.target_loss_2.reset_states()
            self.target_accuracy_2.reset_states()

        return


