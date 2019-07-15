import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_backward
import mnist_forward
import mnist_hyper_parameter as mhp

import os
# log输出级别
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
# 测试图片相对路径
apppath ='full_connect_nn/pic/0.png'
apppathb =r'full_connect_nn\pic\0.png'
    
    
def restor_model(test_pic_ary):
    with tf.Graph().as_default() :
        x=tf.placeholder(tf.float32,[None,mhp.INPUT_NODE])
        y=mnist_forward.forward(x,None)        
        pre_value=tf.argmax(y,1)
        
        # 实例化带有滑动平均值的save
        variable_averages=tf.train.ExponentialMovingAverage(mhp.MOVE_AVERAGE_DECAY)
        variable_to_restore=variable_averages.variables_to_restore()
        save=tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            save_path=mhp.get_save_path()
            if save_path:
                save.restore(sess,save_path)
                pre_value=sess.run(pre_value,feed_dict={x:test_pic_ary})
                return pre_value
            else:
                print("No checkpoint file found")
                return -1

def pre_pic(pic_name):
    img=Image.open(pic_name)
    # 设置图片大小，并消除锯齿
    reim=img.resize((28,28),Image.ANTIALIAS)
    #由于神经网络只支持灰度图，所以将数据转换成灰度图
    im_ary=np.array(reim.convert('L'))
    # 由于神经网络支持的是黑底白字，将白底黑字反色成黑底白字
    threshold = 50
    for i in range(28):
         for j in range(28):
            #  首先反色
             im_ary[i][j]=255-im_ary[i][j]
            #  将像素转成黑色或白色数据，过滤掉噪声
             if (im_ary[i][j]<threshold):
                 im_ary[i][j]=0 
             else:
                 im_ary[i][j]=255
    # 设置数据的形状为满足神经网络的输入特征节点
    nm_ary=im_ary.reshape([1,mhp.INPUT_NODE])
    # 将数据转换成32位浮点数
    nm_ary=nm_ary.astype(np.float32)
    # 将1到255的浮点数转换成0到1的浮点数
    img_ready = np.multiply(nm_ary,1./255.)
    # 返回处理好的数据
    return img_ready


def application():
    test_num = input("input the number of test pictures:")
    test_num = int(test_num)
    for i in range(test_num):
        test_pic=input("the path of thes pictures:")
        test_pic_ary=pre_pic(test_pic)
        pre_value=restor_model(test_pic_ary)
        print("the prediction number is:",pre_value)

def main():
    application()

if __name__=="__main__":
    main()



