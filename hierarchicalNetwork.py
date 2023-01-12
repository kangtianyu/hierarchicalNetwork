# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import hashlib

class HierarchicalNetwork:
    def train(self, x, y, batch_size=64):
        
        # Instantiate an optimizer.
        optimizer = keras.optimizers.SGD(learning_rate=self.lambda1)
        # Instantiate a loss function.
        loss_fn = keras.losses.MeanSquaredError()
#         loss_fn = lambda ey, y: tf.math.square(tf.math.subtract(ey,y))

        train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
        
        for epoch in range(100):
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
     
                
                    
    #                 tf.compat.v1.disable_eager_execution()
                    
    #                 models = tf.map_fn(lambda x:self.f(x),x_batch_train)
                predict = []
                with tf.GradientTape() as tape:
                    tape.watch(self.w)
                    
                    predict = tf.map_fn(lambda x:self.f(x),x_batch_train,dtype = 'float32')
#                     for s in x_batch_train:
#     #                     print(s)
#                         out = self.f(s)
#                         
#                         predict.append(out)
             
    #                     logits = model(s, training=True)  # Logits for this minibatch
             
                        # Compute the loss value for this minibatch.
    
                        # Use the gradient tape to automatically retrieve
                    # the gradients of the trainable variables with respect to the loss.
                    
#                     predict = tf.stack(predict)
                    loss_value = loss_fn(y_batch_train, predict)
                    
                    
                    grads = tape.gradient(loss_value, self.w)
    
                    # Run one step of gradient descent by updating
                    # the value of the variables to minimize the loss.
                    optimizer.apply_gradients([(grads, self.w)])
                print(
                    "."
                )

            # Log every 10 epoches.
            if epoch % 1 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (epoch, float(loss_value))
                )
            
        
    def eval(self, idx):
        if self.f(idx) > 0.5: return 1
        else: return 0
    
    def f(self, idx):

        s = self.getSentenceById(idx)
#         print(s)
        h = self.getHiera(s)
        outputs = self.f_rec(h,s)
        return outputs
    
    def f_rec(self,h,s):
        key = h[0]
        inputs = h[1]
        if not isinstance(h[1], list):
            return self.hnet(1,[key,inputs],s)
        l = []
        for a in inputs:
            l.append(self.f_rec(a,s))
        output = self.act(l)
        return self.hnet(output,key,s)
    
    #activation function
    def act(self,l):
        return tf.math.reduce_sum(tf.math.scalar_mul(self.lambda2,tf.Variable(l)))
    
    def hnet(self,l,key,s):
        key_emb = s['tokens'][key[1]]['embedding']
        cents = self.centroids[key[0]]
        nst_emb = self.nearest_emb(key_emb,cents)
        x = tf.math.reduce_sum(tf.math.multiply(nst_emb,self.w[self.wordIdxMap[key[0]]]))
        return x
        
    def nearest_emb(self,key_emb,cents):
        dis = tf.norm(tf.math.subtract(key_emb,cents[-1]), ord='euclidean')
        idx = -1
        for i in range(len(cents)-1):
            new_dis = tf.norm(tf.math.subtract(key_emb,cents[i]), ord='euclidean')
            if new_dis < dis:
                dis = new_dis
                idx = i;
        return cents[idx]
    
    def getHiera(self,s):
        dic = hashlib.md5(s['text'].encode('utf-8')).hexdigest()
        return self.hiera[dic]
    
    def getSentenceById(self, idx):
        return self.idxtos[idx.numpy()]
    
    
    def __init__(self, centroids,wordIdxMap,IdxWordMap,hiera,idxtos):
        self.centroids = centroids
        self.wordIdxMap = wordIdxMap
        self.IdxWordMap = IdxWordMap
        self.hiera = hiera
        self.idxtos = idxtos
#         print(len(self.centroids),len(list(self.centroids.values())[0][0]))
        self.w = tf.Variable(tf.random.normal([len(self.centroids),len(list(self.centroids.values())[0][0])]))
        
        self.lambda1 = 0.0005
        self.lambda2 = 0.1
        
        
        
        