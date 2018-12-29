#一个拥有一个隐层的简单前制神经网络
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
#mnist数据集相关常数

INPUT_NODE=784      #输入层的节点数，对于mnist数据集，这就相当于图片的像素
OUTPUT_NODE=10      #输出层的节点数，这等同于类别的数量

#配置神经网络参数
LAYER1_NODE=500     #隐藏层节点数
BATCH_SIZE=100      #训练一个batch中的训练数据个数
LEARNING_RATE_BASE=0.8     #基础学习率
LEARNING_RATE_DECAY=0.99   #学习的衰减率
REGULARIZATION_RATE=0.0001 #描述模型复杂度的正则化在损失函数中的系数
TRAINING_STEP=30000        #设置模型的迭代次数
MOVING_AVERAGE_DECAY=0.99  #滑动平均衰减率

#定义一个辅助函数,给定神经网络的所有参数,计算网络的向前传播结果
#定义一个使用RELU激活函数的三层全连接神经网络.通过加入隐藏层实现多层网络结构
#通过RELU激活函数进行去线性化,
def inference(input_tensor,avg_class,weight1,biases1,weight2,biases2):
    #当没有提供滑动平均类时,直接使用参数的当前值.
    if avg_class==None:
        #计算隐层的向前传播结果,并使用激活函数去线性化
        layer1=tf.nn.relu(tf.matmul(input_tensor,weight1)+biases1)
        return tf.matmul(layer1,weight2)+biases2
    else:
        #使用avg_class.average函数去计算得出变量的滑动平均值,然后在计算相应的神经网络结果

        layer1=tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weight1))+avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weight2))+avg_class.average(biases2)

    #模型训练过程
def train(mnist):
    x_ = tf.placeholder(dtype=tf.float32, shape=(None, INPUT_NODE), name='x-input')
    y_ = tf.placeholder(dtype=tf.float32, shape=(None, OUTPUT_NODE), name='y-input')

    # 生成隐层
    weight1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 生成输出层
    weight2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2= tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

    y=inference(x_,None,weight1,biases1,weight2,biases2)

    global_step=tf.Variable(0,trainable=False)  #制定这个变量为不可轮转变量

    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)

    #在所有代表神经网络参数变量上使用滑动平均.其他辅助变量就不再需要(比如global_step),
    #tf.trainable_variables返回的就是图上集合GraphKeys.TRAINABLE_VARIABLES中的元素,这个集合中
    #的元素就是没有指定trainbale=Fasle的参数
    variable_averages_op=variable_averages.apply(tf.trainable_variables())
    average_y=inference(x_,variable_averages,weight1,biases1,weight2,biases2)
    #计算交叉熵作为刻画预测值和真实值之间差距的损失函数,通过使用
    # sparse_softmax_cross_entropy_with_logit函数计算交叉熵
    print(tf.arg_max(y_,1))
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    #计算在当前batch中所有样例的交叉熵均值.
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    # 计算L2正则化损失函数
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #计算模型的正则损失.一般只计算神经边上权重的正则化损失,而不使用偏执项
    regularization=regularizer(weight1)+regularizer(weight2)
    #总损失等于交叉熵损失和正侧化损失的和
    loss=regularization+cross_entropy_mean
    # 设置指数衰减学习率
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,
                                             mnist.train.num_examples,LEARNING_RATE_DECAY)
    #使用优化算法对损失函数进行优化
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step)
    #在训练模型时,每过一遍数据集需要通过反向传播来更新神经网络中的参数
    #同时需要更新每一个参数的滑动平均值
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op=tf.no_op(name='train')
    correct_prediction=tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    #tf.cast进行数值的格式转化
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    #初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        validate_feed={x_:mnist.validation.images,y_:mnist.validation.labels}

        test_feed={x_:mnist.test.images,y_:mnist.test.labels}
        for i in range(TRAINING_STEP):
            if i%1000==0:
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                print('在训练%d次后,训练模型的精确度为:%g'%(i,validate_acc))
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x_:xs,y_:ys})
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print('模型的精确度为:%g'%test_acc)

#主程序入口
def main(argv=None):
        mnist=input_data.read_data_sets('D:\\fengxu\\PycharmProjects\\untitled\\神经网络\\神经网络\\卷积网络\\MNIST_DATA',one_hot=True)
        train(mnist)

if __name__ == '__main__':
        tf.app.run()




