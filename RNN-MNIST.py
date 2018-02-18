#Code by Sridhar.T
#BSD License
#First download MNIST Data Set from YannLecun's webpage!
#MNIST Data set, image is 28x28
#Each row of the image is taken as a time step, hence sequence consists of 28 timesteps
#At the end of the sequence, network has to output the digit, cross entropy loss used
#Uncomment out plots line in end if needed
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)
#fig1=plt.figure()
num_hidden1=300
num_hidden_lstm=200 #LSTM neurons
num_classes=10
input_dim=28
batch_size=100
num_epochs=5000
seq_length=28 #this is the maximum length of the sequence.
epoch=np.arange(num_epochs)
epoch=[]
accur=[]

x=tf.placeholder('float32',[None,seq_length,input_dim]) #None is batch size
y=tf.placeholder('float32',[None,num_classes])

W=tf.Variable(tf.random_normal([input_dim,num_hidden_lstm]))
W = tf.reshape(W, (1, input_dim,num_hidden_lstm)) s
a=tf.shape(x)[0]
W = tf.tile(W, [a, 1, 1]) #Since input is 3d, and normal full conectted net is a vector, we need to tile appropriately.
act1=tf.batch_matmul(x,W) #multiply (think this changed in most recent of tensorflow, do verify).
cell=tf.nn.rnn_cell.BasicLSTMCell(num_hidden_lstm, state_is_tuple=True) #basic lstm cell.
out, h_state = tf.nn.dynamic_rnn(cell,act1, dtype=tf.float32)   #the output as well as the hidden state at the end of the sequence
val = tf.transpose(out, [1, 0, 2]) 
last = tf.gather(val, int(val.get_shape()[0]) - 1) #getting only the last output at the sequence- i.e time tn. we don't need time tn-1 etc to classify.
W1=tf.Variable(tf.random_normal([num_hidden_lstm,num_classes]))
bias1=tf.Variable(tf.random_normal([num_classes]))
final_act=tf.add(tf.matmul(last,W1),bias1)

loss=tf.nn.softmax_cross_entropy_with_logits(final_act,y) # softmax loss
optimizer=tf.train.AdamOptimizer().minimize(loss)
init_op=tf.initialize_all_variables()

sess=tf.InteractiveSession()
sess.run(init_op)
saver=tf.train.Saver()
for k in range(num_epochs):
    xbatch,ybatch=mnist.train.next_batch(batch_size)
    xbatch=xbatch.reshape(batch_size,seq_length,input_dim)
    #saver.restore(sess,"./model/weights60.ckpt")
    sess.run(optimizer,feed_dict={x:xbatch,y:ybatch})
    if k%25==0:
        
        correct_prediction = tf.equal(tf.argmax(final_act, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
        xtest,ytest=mnist.test.next_batch(1000)
        xtest=xtest.reshape(1000,seq_length,input_dim)
        accur.append(accuracy.eval({x:xtest,y:ytest}))
        epoch.append(k)
        #plt.plot(epoch,accur)
        #plt.show()
        print('Epoch %d completed with accuracy %.2f'%(k+1,accuracy.eval({x:xtest,y:ytest})*100))
        #np.savetxt(sess.run())
        #save_path=saver.save(sess,'./model/weights%d.ckpt' %(k))
   # print('done with epoch %d' %(k))
    

