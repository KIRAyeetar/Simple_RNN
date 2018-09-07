from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import tensorflow as tf


def run():
    # =================
    #  read iris data
    # =================
    iris = load_iris()
    iris_data = pd.DataFrame(iris.data)
    iris_data['target'] = iris.target
    iris_data = iris_data.sample(frac=1, random_state=10)
    iris_data = iris_data.sample(frac=1)

    train = iris_data.head(120)
    test = iris_data.tail(30)

    train['tar_0'] = [1 if i == 0 else 0 for i in train['target']]
    train['tar_1'] = [1 if i == 1 else 0 for i in train['target']]
    train['tar_2'] = [1 if i == 2 else 0 for i in train['target']]
    test['tar_0'] = [1 if i == 0 else 0 for i in test['target']]
    test['tar_1'] = [1 if i == 1 else 0 for i in test['target']]
    test['tar_2'] = [1 if i == 2 else 0 for i in test['target']]

    train_x = np.array(train.loc[:, [0, 1, 2, 3]])
    test_x = np.array(test.loc[:, [0, 1, 2, 3]])
    train_y = np.array(train.loc[:, ['tar_0', 'tar_1', 'tar_2']])
    test_y = np.array(test.loc[:, ['tar_0', 'tar_1', 'tar_2']])

    # =================
    #  start alchemy
    # =================
    # 喂入特征的长度
    INPUT_LEN = 4
    # 输出结果的长度
    OUTPUT_LEN = 3
    # BATCH_SIZE的大小
    BATCH_SIZE = 120
    # 循环训练多少次
    STEPS = 1000
    # 基准学习率
    LEARNING_RATE_BASE = 0.01

    # 下面两个参数解释下，特征有4维，看你怎么划分他们
    # 比如你认为 前两个特征和后两个特征有序列上的关系，你就划分成2x2的序列
    # 如果就是 四个毫无关联 的特征，那就直接 1x4就行
    SEQUENCE_LEN = 4
    FRAME_LEN = 1

    # 隐藏层神经元数
    HIDDEN_LEN = 100
    # 设置几层RNN
    RNN_LAYER_LEN = 2
    # drop_out的参数
    OUT_KEEP_PROB = 1
    # global_step =

    # 给要喂的训练集特征创建占位符
    x = tf.placeholder(tf.float32, [None, INPUT_LEN])
    # 给要喂的训练集标签创建占位符
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_LEN])
    # 定义权重和常数项，这里是输出层的
    # 那输入层的呢？后面直接用方法创建的CELL内部会自行解决
    w = tf.Variable(tf.truncated_normal(shape=[HIDDEN_LEN,OUTPUT_LEN], seed=1))
    b = tf.Variable(tf.zeros(shape=[OUTPUT_LEN]))

    # 把输入的数据转换格式
    input_data = tf.reshape(x, shape=[-1, SEQUENCE_LEN, FRAME_LEN])
    # 创建基础的CELL数组，有几层数组里就有几个CELL
    # 再提一句，CELL里的权重和常数项会自动适配的，不用多担心，虽然我还没试过自己设置
    # 为什么不在MultiRNNCell方法再创建数组呢？因为实测会报错... ...
    rnn_cell = [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_LEN, state_is_tuple=True) for i in range(RNN_LAYER_LEN)]
    # DROP_OUT, 数据量小就算了
    rnn_cell = [tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=1.0, output_keep_prob=OUT_KEEP_PROB) for cell in rnn_cell]
    # 依照数组里的CELL创建多层RNN
    rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_cell, state_is_tuple=True)
    # 输出CELL里的结果，也就是输出层的结果
    output_data, states = tf.nn.dynamic_rnn(rnn_cell, input_data, dtype=tf.float32)
    # 把输出层的结果用softmax转换下
    y = tf.nn.softmax(tf.matmul(output_data[:,-1,:], w)+b, 1)

    # 定义LOSS
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    # ADAM最小化LOSS
    train_step = tf.train.AdamOptimizer(LEARNING_RATE_BASE).minimize(loss)
    # 定义准确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(test_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    with tf.Session() as sess:
        # 全局初始化
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 开跑了开跑了
        for i in range(STEPS):
            # 取得一个BATCH_SIZE的训练数据
            start = (i * BATCH_SIZE) % len(train_x)
            end = start + BATCH_SIZE
            sess.run(train_step, feed_dict={x: train_x[start: end, :],
                                            y_: train_y[start: end, :]})
            # 每100次打印相关的loss和acc
            if i % 100 == 0:
                cross_loss = sess.run(loss, feed_dict={x: train_x[start: end, :], y_: train_y[start: end, :]})
                acc = sess.run(accuracy, feed_dict={x: test_x})
                print(i, cross_loss, acc)


if __name__ == '__main__':
    run()
