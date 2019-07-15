import os
import tensorflow as tf
import numpy as np
from PIL import Image

image_train_path='./mnist_tata_jpg/mnist_train_jpg_60000/'
label_train_path='./mnist_tata_jpg/mnist_train_jpg_60000.txt'
tfrecord_train='./data/mnist_test.tfrecords'

image_test_path='./mnist_tata_jpg/mnist_train_jpg_10000/'
label_test_path='./mnist_tata_jpg/mnist_train_jpg_10000.txt'
tfrecord_test='./data/mnist_test.tfrecords'

data_path='./data'
resize_height=28
resize_width=28


def write_tfrecord(tfrecorde_name,image_path,label_path):
    writer=tf.python_io.TFRecordWriter(tfrecorde_name)
    num_pic=0
    f=open(label_path,'r')
    contents=f.readlines()
    f.close()
    for content in contents:
        value=content.split()
        img_path=image_path+value[0]
        img=Image.open(img_path)
        img_raw=img.tobytes()
        labels=[0]*10
        labels[int(value[1])]=1
        example=tf.train.Example(features=tf.train.Features(feature={
            'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'label':tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
        }))
        writer.write(example.SerializeToString())
        num_pic+=1
        print('the number of picture:',num_pic)
    writer.close()
    print('write tfrecord successful')


def generate_tfrecord():
    is_exists=os.path.exists(data_path)
    if not is_exists:
        os.makedirs(data_path)
        print('the directory was created successfully')
    else:
        print('the directory alread exists')
    write_tfrecord(tfrecord_train,image_train_path,label_train_path)
    write_tfrecord(tfrecord_test,image_test_path,label_test_path)

def read_tfrecord(tfrecord_path):
    filename_queue=tf.train.string_input_producer([tfrecord_path])
    reader=tf.TFRecordReader()
    _,serialized_example=reader.read(filename_queue)
    features=tf.parse_single_example(serialized_example,
    features={
        'label':tf.FixedLenFeature([10],tf.int64),
        'img_raw':tf.FixedLenFeature([],tf.string)
    })
    img=tf.decode_raw(features['img_raw'],tf.uint8)
    img.set_shape([784])
    img=tf.cast(img,tf.float32)*(1./255)
    label=tf.cast(features['label'],tf.float32)
    return img,label


def get_tfrecord(num,is_train=True):
    if is_train:
        tfrecord_path=tfrecord_train
    else:
        tfrecord_path=tfrecord_test
    
    img,label=read_tfrecord(tfrecord_path)
    img_batch,label_batch=tf.train.shuffle_batch([img,label],
    batch_size=num,
    num_threads=2,
    capacity=1000,
    min_after_dequeue=700)
    return img_batch,label_batch


def main():
    generate_tfrecord()

if __name__ =='__main__':
    main()


