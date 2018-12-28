#加载数据
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_DATA/',one_hot=True)
# print(mnist.train.num_examples,mnist.validation.num_examples,mnist.test.num_examples,mnist.train.labels[0])

sess=tf.InteractiveSession()

def weight_variable(shape):
    weight=tf.Variable(initial_value=tf.truncated_normal(shape=shape,stddev=0.1),name='weight')
    return weight

def bias_variable(shape):
    biase=tf.Variable(initial_value=tf.constant(shape=shape,value=0.1),name='bias')
    return biase

def conv_op(in_tensor,kernel,strides=[1,1,1,1],padding='SAME'):
    conv_out=tf.nn.conv2d(in_tensor,kernel,strides,padding)
    return conv_out

def max_pool(in_tensor,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',):
    max_pool_output=tf.nn.max_pool(in_tensor,ksize=ksize,strides=strides,padding=padding,name='max_pool')
    return max_pool_output
def simple_cnn():
    x=tf.placeholder(tf.float32,[None,784])
    y_=tf.placeholder(tf.float32,[None,10])
    x_image=tf.reshape(tensor=x,shape=[-1,28,28,1])
    #结构
    #卷积层
    w1=[5,5,1,32]
    b1=[32]
    w2=[5,5,32,64]
    b2=[64]
    wfc1=[7*7*64,1024]
    bfc1=[1024]
    wfc2=[1024,10]
    bfc2=[10]

    #第一隐层
    w_conv1=weight_variable(w1)
    b_conv1=bias_variable(b1)
    h_conv1=tf.nn.relu(conv_op(x_image,w_conv1)+b_conv1)
    h_pool1=max_pool(h_conv1)

    #第二隐层
    w_conv2=weight_variable(w2)
    b_conv2=bias_variable(b2)
    h_conv2=tf.nn.relu(conv_op(h_pool1,w_conv2)+b_conv2)
    h_pool2=max_pool(h_conv2)

    #第三隐层
    h_pool2_flat=tf.reshape(tensor=h_pool2,shape=[-1,7*7*64])
    W_fc1=weight_variable(wfc1)
    b_fc1=bias_variable(bfc1)
    h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
    keep_prob=tf.placeholder(tf.float32)
    h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

    #第四隐层
    W_fc2=weight_variable(wfc2)
    b_fc2=bias_variable(bfc2)
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

    #loss function
    cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
    train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    #estimate accuarcy
    correct_prediction=tf.equal(tf.arg_max(y_conv,1),tf.arg_max(y_,1))
    accurate=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    tf.global_variables_initializer().run()
    for i in range(20000):
        batch=mnist.train.next_batch(50)
        if i%100==0:
            train_accurate=accurate.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
            print('step %d,training accuracy %g'%(i,train_accurate))
        train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
        print('test accuarcy %g'%accurate.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:0.1}))
        return


if __name__ == '__main__':
    simple_cnn()
