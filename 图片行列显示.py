from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
mnist=input_data.read_data_sets('D:\\fengxu\\PycharmProjects\\untitled\\神经网络\\神经网络\\卷积神经网络\\MNIST_DATA',one_hot=False)
#训练数据大小
train_nums=mnist.train.num_examples
# print(train_nums)
#验证数据大小
validation_num=mnist.validation.num_examples
# print(mnist.validation.num_examples)
train_data=mnist.train.images
# print(train_data[0].shape)    #一张图片的大小
batch_size=12
xs,ys=mnist.train.next_batch(batch_size)
# print(ys.shape,xs.shape)

#图像显示
def plot_gallery(images,title,h,w,n_row=10,n_col=15):
    plt.figure(figsize=(2*n_col,2.2*n_row),dpi=144)
    plt.subplots_adjust(bottom=0,left=.01,right=.99,top=.9,hspace=.01)
    for i in range(len(images)):
        plt.subplot(n_row,n_col,i+1)
        plt.imshow(images[i].reshape((h,w)))
        plt.title(title[i])
        plt.axis('off')
    plt.show()
n_row=2
n_col=6
# print(xs.shape,ys.shape)
train_data_titles=['c%d'%i for i in ys]
plot_gallery(xs,train_data_titles,28,28,n_row,n_col)
