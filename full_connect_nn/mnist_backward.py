import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os
import mnist_hyper_parameter as mhp


def backwarad(mnist):
    """
    反向传播：训练网络，优化网络参数，提高模型准确性
      训练网络模型时常将正则化，指数衰减学习和滑动平均这三个方法做为模型优化方法      
    """
    # 数据集占位
    x=tf.placeholder(tf.float32,[None,mhp.INPUT_NODE])
    # 标准答案点位
    y_=tf.placeholder(tf.float32,[None,mhp.OUTPUT_NODE])
    # 计算的预测结果，由forward()实现前向传播的网络结构
    y=mnist_forward.forward(x,mhp.REGULARIZER)
    # 训练轮数，设置为不可训练型参数
    global_step=tf.Variable(0,trainable=False)

    # 正则化表示为：
    # 首先，定义损失函数：计算预测结果与标准答案的损失值。有三种形式1:MSE,2:CE(交叉熵)，3自宝义：y与y_的差距
    ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cem=tf.reduce_mean(ce)
    # 其次，总损失值为预测结果与标准答案的损失值加上正则化项
    loss = cem + tf.add_n(tf.get_collection('losses'))

    # 指数衰减学习率
    learning_rate=tf.train.exponential_decay(
        mhp.LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/mhp.BATCH_SIZE,
        mhp.LEARNING_RATE_DECAY,
        staircase=True
    )

    train_step=mhp.get_optimizer(learning_rate).minimize(loss,global_step=global_step)
    # train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    # train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)

    # 滑动平均,学习率与指数平均学习率中的global_step为同一参数
    ema =tf.train.ExponentialMovingAverage(mhp.MOVE_AVERAGE_DECAY,global_step)
    ema_op=ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step,ema_op]):
        train_op=tf.no_op(name='train')

    saver=tf.train.Saver()

    with tf.Session() as sess:
        init_op=tf.global_variables_initializer()
        sess.run(init_op)       
        
        save_path=mhp.get_save_path()
        if save_path :
            saver.restore(sess,save_path)

        for i in range(mhp.STEPS):
            xs,ys=mnist.train.next_batch(mhp.BATCH_SIZE)
            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            if i%1000==0:
                print("after %d training step(s),loss on training batch is %g."%(step,loss_value))
                saver.save(sess,os.path.join(mhp.MODEL_SAVE_PATH,mhp.MODEL_NAME),global_step=global_step)

def main():
    mnist=input_data.read_data_sets(mhp.MNIST_DATA_PATH,one_hot=True)
    backwarad(mnist)

if __name__=='__main__':
    main()