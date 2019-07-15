import tensorflow as tf
INPUT_NODE=784
OUTPUT_NODE=10
LAYER_NODE=500


BATCH_SIZE=100
LEARNING_RATE_BASE=0.01
LEARNING_RATE_DECAY=0.99
REGULARIZER=0.0001
STEPS=50000
MOVE_AVERAGE_DECAY=0.99
MNIST_DATA_PATH="./MNIST_data/"
MODEL_NAME="mnist_model"
PIC_PATH="full_connect_nn/pic/0.png"

MNIST_DATA_PATH_FOR_ADAM="./full_connect_nn/model/adam/"
MNIST_DATA_PATH_FOR_GRADIENT_DES="./full_connect_nn/model/gradient_des/"
MODEL_SAVE_PATH=MNIST_DATA_PATH_FOR_ADAM

def get_optimizer(lear_rate):
    """
    由于优化器不同，计算模型参数的过程也不相同，所以采取模型分开保存
    """
    if MODEL_SAVE_PATH == MNIST_DATA_PATH_FOR_ADAM:
        return tf.train.AdamOptimizer(lear_rate)
    else:
        return tf.train.GradientDescentOptimizer(lear_rate)

def get_save_path():
    # 实现断点续训
    ckpt=tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        return ckpt.model_checkpoint_path
    return None
    