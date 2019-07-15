import time
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward
import mnist_hyper_parameter as mhp

TEST_INTERVAL_SECE=5

def test(mnist):    
    with tf.Graph().as_default():
        x=tf.placeholder(tf.float32,[None,mhp.INPUT_NODE])
        y_=tf.placeholder(tf.float32,[None,mhp.OUTPUT_NODE])
        y=mnist_forward.forward(x,None)

        ema = tf.train.ExponentialMovingAverage(mhp.MOVE_AVERAGE_DECAY)
        ema_restore=ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction =tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        while True:
            with tf.Session() as sess:
                save_path =mhp.get_save_path()
                if save_path:
                    saver.restore(sess,save_path)
                    global_step=save_path.split('/')[-1].split('-')[-1]
                    accuracy_score=sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
                    print("after %s training step(s), test accuracy=%g"%(global_step,accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(TEST_INTERVAL_SECE)

def main():
    mnist = input_data.read_data_sets(mhp.MNIST_DATA_PATH,one_hot=True)
    test(mnist)

if __name__=='__main__':
    main()