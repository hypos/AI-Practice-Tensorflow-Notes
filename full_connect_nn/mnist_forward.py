import tensorflow as tf
import mnist_hyper_parameter as mhp


def get_weight(shape,regularizer):    
    w=tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer != None : tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b=tf.Variable(tf.zeros(shape))
    return b

def forward(x,regularizer):
    w1=get_weight([mhp.INPUT_NODE,mhp.LAYER_NODE],regularizer)
    b1=get_bias([mhp.LAYER_NODE])
    y1=tf.nn.relu(tf.matmul(x,w1)+b1)

    w2=get_weight([mhp.LAYER_NODE,mhp.OUTPUT_NODE],regularizer)
    b2=get_bias([mhp.OUTPUT_NODE])
    y2=tf.matmul(y1,w2)+b2
    return y2


