
# coding: utf-8

# In[ ]:


# load modules and set paths
import numpy as np
from keras.models import Sequential,load_model, Model
from keras.layers import Input,Dense,Conv2D,CuDNNLSTM,GaussianNoise,GaussianDropout,Reshape,Bidirectional,Lambda,Conv2DTranspose,Permute,Subtract
import keras.layers as L
from keras import optimizers
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import LearningRateScheduler,ModelCheckpoint
from keras import backend as K
import h5py
import math
base_dir=''
dataset_name = base_dir+'KITTI_set_sph1.h5'
model_name=base_dir+'ChipNet_KITTI2-10layer-float.h5'
weights_name=base_dir+'ChipNet_KITTI2-10layer-float-weights.h5'
pretrain_name=base_dir+'ChipNet_KITTI2-10layer-float-weights.h5'


# In[ ]:


# load dataset and seperate
fid = h5py.File(dataset_name, 'r')
list_length=fid['img'].shape[0]
print ('{:6d} samples in the dataset'.format(list_length))

train_list_length = list_length-179


x_train = HDF5Matrix(dataset_name, 'img', start=0, end=train_list_length)
y_train = HDF5Matrix(dataset_name, 'gt' , start=0, end=train_list_length)


x_val = HDF5Matrix(dataset_name, 'img', start=train_list_length, end=list_length)
y_val = HDF5Matrix(dataset_name, 'gt' , start=train_list_length, end=list_length)


print "shape of x_train: {:.30s}".format(np.array(x_train.shape)) 
print "shape of y_train: {:.30s}".format(np.array(y_train.shape))

print "shape of x_val:   {:.30s}".format(np.array(x_val.shape)) 
print "shape of y_val:   {:.30s}".format(np.array(y_val.shape)) 


# In[ ]:


from PIL import Image
Image.fromarray(np.uint8(y_val[1]*255))


# In[ ]:



inputs = Input(shape=(64,180,14))
x = Conv2D(64,(5,5),name='conv0',padding='same')(inputs)
x = L.Activation('tanh')(x)


# conv1
x1 = Lambda(lambda x : x[:,:,:, 0:16])(x)
x2 = Lambda(lambda x : x[:,:,:,16:32])(x)
x3 = Lambda(lambda x : x[:,:,:,32:48])(x)
x4 = Lambda(lambda x : x[:,:,:,48:64])(x)

x1w= Conv2D(16,(3,3),name='conv1_x1w',padding='same')(x1)
x1v= Conv2D(16,(3,3),name='conv1_x1v',padding='same',dilation_rate=2)(x1)
x1 = L.Add()([x1,x1w,x1v])
x1 = L.Activation('tanh')(x1)

x2w= Conv2D(16,(3,3),name='conv1_x2w',padding='same')(x2)
x2v= Conv2D(16,(3,3),name='conv1_x2v',padding='same',dilation_rate=2)(x2)
x2 = L.Add()([x2,x2w,x2v])
x2 = L.Activation('tanh')(x2)

x3w= Conv2D(16,(3,3),name='conv1_x3w',padding='same')(x3)
x3v= Conv2D(16,(3,3),name='conv1_x3v',padding='same',dilation_rate=2)(x3)
x3 = L.Add()([x3,x3w,x3v])
x3 = L.Activation('tanh')(x3)

x4w= Conv2D(16,(3,3),name='conv1_x4w',padding='same')(x4)
x4v= Conv2D(16,(3,3),name='conv1_x4v',padding='same',dilation_rate=2)(x4)
x4 = L.Add()([x4,x4w,x4v])
x4 = L.Activation('tanh')(x4)

x = L.Concatenate()([x1,x2,x3,x4])
x = Conv2D(64,(1,1),name='conv1_concat',padding='same')(x)
x = L.Activation('tanh')(x)

# conv2
x1 = Lambda(lambda x : x[:,:,:, 0:16])(x)
x2 = Lambda(lambda x : x[:,:,:,16:32])(x)
x3 = Lambda(lambda x : x[:,:,:,32:48])(x)
x4 = Lambda(lambda x : x[:,:,:,48:64])(x)

x1w= Conv2D(16,(3,3),name='conv2_x1w',padding='same')(x1)
x1v= Conv2D(16,(3,3),name='conv2_x1v',padding='same',dilation_rate=2)(x1)
x1 = L.Add()([x1,x1w,x1v])
x1 = L.Activation('tanh')(x1)

x2w= Conv2D(16,(3,3),name='conv2_x2w',padding='same')(x2)
x2v= Conv2D(16,(3,3),name='conv2_x2v',padding='same',dilation_rate=2)(x2)
x2 = L.Add()([x2,x2w,x2v])
x2 = L.Activation('tanh')(x2)

x3w= Conv2D(16,(3,3),name='conv2_x3w',padding='same')(x3)
x3v= Conv2D(16,(3,3),name='conv2_x3v',padding='same',dilation_rate=2)(x3)
x3 = L.Add()([x3,x3w,x3v])
x3 = L.Activation('tanh')(x3)

x4w= Conv2D(16,(3,3),name='conv2_x4w',padding='same')(x4)
x4v= Conv2D(16,(3,3),name='conv2_x4v',padding='same',dilation_rate=2)(x4)
x4 = L.Add()([x4,x4w,x4v])
x4 = L.Activation('tanh')(x4)

x = L.Concatenate()([x1,x2,x3,x4])
x = Conv2D(64,(1,1),name='conv2_concat',padding='same')(x)
x = L.Activation('tanh')(x)


# conv3
x1 = Lambda(lambda x : x[:,:,:, 0:16])(x)
x2 = Lambda(lambda x : x[:,:,:,16:32])(x)
x3 = Lambda(lambda x : x[:,:,:,32:48])(x)
x4 = Lambda(lambda x : x[:,:,:,48:64])(x)

x1w= Conv2D(16,(3,3),name='conv3_x1w',padding='same')(x1)
x1v= Conv2D(16,(3,3),name='conv3_x1v',padding='same',dilation_rate=2)(x1)
x1 = L.Add()([x1,x1w,x1v])
x1 = L.Activation('tanh')(x1)

x2w= Conv2D(16,(3,3),name='conv3_x2w',padding='same')(x2)
x2v= Conv2D(16,(3,3),name='conv3_x2v',padding='same',dilation_rate=2)(x2)
x2 = L.Add()([x2,x2w,x2v])
x2 = L.Activation('tanh')(x2)

x3w= Conv2D(16,(3,3),name='conv3_x3w',padding='same')(x3)
x3v= Conv2D(16,(3,3),name='conv3_x3v',padding='same',dilation_rate=2)(x3)
x3 = L.Add()([x3,x3w,x3v])
x3 = L.Activation('tanh')(x3)

x4w= Conv2D(16,(3,3),name='conv3_x4w',padding='same')(x4)
x4v= Conv2D(16,(3,3),name='conv3_x4v',padding='same',dilation_rate=2)(x4)
x4 = L.Add()([x4,x4w,x4v])
x4 = L.Activation('tanh')(x4)

x = L.Concatenate()([x1,x2,x3,x4])
x = Conv2D(64,(1,1),name='conv3_concat',padding='same')(x)
x = L.Activation('tanh')(x)

# conv4
x1 = Lambda(lambda x : x[:,:,:, 0:16])(x)
x2 = Lambda(lambda x : x[:,:,:,16:32])(x)
x3 = Lambda(lambda x : x[:,:,:,32:48])(x)
x4 = Lambda(lambda x : x[:,:,:,48:64])(x)

x1w= Conv2D(16,(3,3),name='conv4_x1w',padding='same')(x1)
x1v= Conv2D(16,(3,3),name='conv4_x1v',padding='same',dilation_rate=2)(x1)
x1 = L.Add()([x1,x1w,x1v])
x1 = L.Activation('tanh')(x1)

x2w= Conv2D(16,(3,3),name='conv4_x2w',padding='same')(x2)
x2v= Conv2D(16,(3,3),name='conv4_x2v',padding='same',dilation_rate=2)(x2)
x2 = L.Add()([x2,x2w,x2v])
x2 = L.Activation('tanh')(x2)

x3w= Conv2D(16,(3,3),name='conv4_x3w',padding='same')(x3)
x3v= Conv2D(16,(3,3),name='conv4_x3v',padding='same',dilation_rate=2)(x3)
x3 = L.Add()([x3,x3w,x3v])
x3 = L.Activation('tanh')(x3)

x4w= Conv2D(16,(3,3),name='conv4_x4w',padding='same')(x4)
x4v= Conv2D(16,(3,3),name='conv4_x4v',padding='same',dilation_rate=2)(x4)
x4 = L.Add()([x4,x4w,x4v])
x4 = L.Activation('tanh')(x4)

x = L.Concatenate()([x1,x2,x3,x4])
x = Conv2D(64,(1,1),name='conv4_concat',padding='same')(x)
x = L.Activation('tanh')(x)

# conv5
x1 = Lambda(lambda x : x[:,:,:, 0:16])(x)
x2 = Lambda(lambda x : x[:,:,:,16:32])(x)
x3 = Lambda(lambda x : x[:,:,:,32:48])(x)
x4 = Lambda(lambda x : x[:,:,:,48:64])(x)

x1w= Conv2D(16,(3,3),name='conv5_x1w',padding='same')(x1)
x1v= Conv2D(16,(3,3),name='conv5_x1v',padding='same',dilation_rate=2)(x1)
x1 = L.Add()([x1,x1w,x1v])
x1 = L.Activation('tanh')(x1)

x2w= Conv2D(16,(3,3),name='conv5_x2w',padding='same')(x2)
x2v= Conv2D(16,(3,3),name='conv5_x2v',padding='same',dilation_rate=2)(x2)
x2 = L.Add()([x2,x2w,x2v])
x2 = L.Activation('tanh')(x2)

x3w= Conv2D(16,(3,3),name='conv5_x3w',padding='same')(x3)
x3v= Conv2D(16,(3,3),name='conv5_x3v',padding='same',dilation_rate=2)(x3)
x3 = L.Add()([x3,x3w,x3v])
x3 = L.Activation('tanh')(x3)

x4w= Conv2D(16,(3,3),name='conv5_x4w',padding='same')(x4)
x4v= Conv2D(16,(3,3),name='conv5_x4v',padding='same',dilation_rate=2)(x4)
x4 = L.Add()([x4,x4w,x4v])
x4 = L.Activation('tanh')(x4)

x = L.Concatenate()([x1,x2,x3,x4])
x = Conv2D(64,(1,1),name='conv5_concat',padding='same')(x)
x = L.Activation('tanh')(x)

# conv6
x1 = Lambda(lambda x : x[:,:,:, 0:16])(x)
x2 = Lambda(lambda x : x[:,:,:,16:32])(x)
x3 = Lambda(lambda x : x[:,:,:,32:48])(x)
x4 = Lambda(lambda x : x[:,:,:,48:64])(x)

x1w= Conv2D(16,(3,3),name='conv6_x1w',padding='same')(x1)
x1v= Conv2D(16,(3,3),name='conv6_x1v',padding='same',dilation_rate=2)(x1)
x1 = L.Add()([x1,x1w,x1v])
x1 = L.Activation('tanh')(x1)

x2w= Conv2D(16,(3,3),name='conv6_x2w',padding='same')(x2)
x2v= Conv2D(16,(3,3),name='conv6_x2v',padding='same',dilation_rate=2)(x2)
x2 = L.Add()([x2,x2w,x2v])
x2 = L.Activation('tanh')(x2)

x3w= Conv2D(16,(3,3),name='conv6_x3w',padding='same')(x3)
x3v= Conv2D(16,(3,3),name='conv6_x3v',padding='same',dilation_rate=2)(x3)
x3 = L.Add()([x3,x3w,x3v])
x3 = L.Activation('tanh')(x3)

x4w= Conv2D(16,(3,3),name='conv6_x4w',padding='same')(x4)
x4v= Conv2D(16,(3,3),name='conv6_x4v',padding='same',dilation_rate=2)(x4)
x4 = L.Add()([x4,x4w,x4v])
x4 = L.Activation('tanh')(x4)

x = L.Concatenate()([x1,x2,x3,x4])
x = Conv2D(64,(1,1),name='conv6_concat',padding='same')(x)
x = L.Activation('tanh')(x)


# conv7
x1 = Lambda(lambda x : x[:,:,:, 0:16])(x)
x2 = Lambda(lambda x : x[:,:,:,16:32])(x)
x3 = Lambda(lambda x : x[:,:,:,32:48])(x)
x4 = Lambda(lambda x : x[:,:,:,48:64])(x)

x1w= Conv2D(16,(3,3),name='conv7_x1w',padding='same')(x1)
x1v= Conv2D(16,(3,3),name='conv7_x1v',padding='same',dilation_rate=2)(x1)
x1 = L.Add()([x1,x1w,x1v])
x1 = L.Activation('tanh')(x1)

x2w= Conv2D(16,(3,3),name='conv7_x2w',padding='same')(x2)
x2v= Conv2D(16,(3,3),name='conv7_x2v',padding='same',dilation_rate=2)(x2)
x2 = L.Add()([x2,x2w,x2v])
x2 = L.Activation('tanh')(x2)

x3w= Conv2D(16,(3,3),name='conv7_x3w',padding='same')(x3)
x3v= Conv2D(16,(3,3),name='conv7_x3v',padding='same',dilation_rate=2)(x3)
x3 = L.Add()([x3,x3w,x3v])
x3 = L.Activation('tanh')(x3)

x4w= Conv2D(16,(3,3),name='conv7_x4w',padding='same')(x4)
x4v= Conv2D(16,(3,3),name='conv7_x4v',padding='same',dilation_rate=2)(x4)
x4 = L.Add()([x4,x4w,x4v])
x4 = L.Activation('tanh')(x4)

x = L.Concatenate()([x1,x2,x3,x4])
x = Conv2D(64,(1,1),name='conv7_concat',padding='same')(x)
x = L.Activation('tanh')(x)


# conv8
x1 = Lambda(lambda x : x[:,:,:, 0:16])(x)
x2 = Lambda(lambda x : x[:,:,:,16:32])(x)
x3 = Lambda(lambda x : x[:,:,:,32:48])(x)
x4 = Lambda(lambda x : x[:,:,:,48:64])(x)

x1w= Conv2D(16,(3,3),name='conv8_x1w',padding='same')(x1)
x1v= Conv2D(16,(3,3),name='conv8_x1v',padding='same',dilation_rate=2)(x1)
x1 = L.Add()([x1,x1w,x1v])
x1 = L.Activation('tanh')(x1)

x2w= Conv2D(16,(3,3),name='conv8_x2w',padding='same')(x2)
x2v= Conv2D(16,(3,3),name='conv8_x2v',padding='same',dilation_rate=2)(x2)
x2 = L.Add()([x2,x2w,x2v])
x2 = L.Activation('tanh')(x2)

x3w= Conv2D(16,(3,3),name='conv8_x3w',padding='same')(x3)
x3v= Conv2D(16,(3,3),name='conv8_x3v',padding='same',dilation_rate=2)(x3)
x3 = L.Add()([x3,x3w,x3v])
x3 = L.Activation('tanh')(x3)

x4w= Conv2D(16,(3,3),name='conv8_x4w',padding='same')(x4)
x4v= Conv2D(16,(3,3),name='conv8_x4v',padding='same',dilation_rate=2)(x4)
x4 = L.Add()([x4,x4w,x4v])
x4 = L.Activation('tanh')(x4)

x = L.Concatenate()([x1,x2,x3,x4])
x = Conv2D(64,(1,1),name='conv8_concat',padding='same')(x)
x = L.Activation('tanh')(x)


# conv9
x1 = Lambda(lambda x : x[:,:,:, 0:16])(x)
x2 = Lambda(lambda x : x[:,:,:,16:32])(x)
x3 = Lambda(lambda x : x[:,:,:,32:48])(x)
x4 = Lambda(lambda x : x[:,:,:,48:64])(x)

x1w= Conv2D(16,(3,3),name='conv9_x1w',padding='same')(x1)
x1v= Conv2D(16,(3,3),name='conv9_x1v',padding='same',dilation_rate=2)(x1)
x1 = L.Add()([x1,x1w,x1v])
x1 = L.Activation('tanh')(x1)

x2w= Conv2D(16,(3,3),name='conv9_x2w',padding='same')(x2)
x2v= Conv2D(16,(3,3),name='conv9_x2v',padding='same',dilation_rate=2)(x2)
x2 = L.Add()([x2,x2w,x2v])
x2 = L.Activation('tanh')(x2)

x3w= Conv2D(16,(3,3),name='conv9_x3w',padding='same')(x3)
x3v= Conv2D(16,(3,3),name='conv9_x3v',padding='same',dilation_rate=2)(x3)
x3 = L.Add()([x3,x3w,x3v])
x3 = L.Activation('tanh')(x3)

x4w= Conv2D(16,(3,3),name='conv9_x4w',padding='same')(x4)
x4v= Conv2D(16,(3,3),name='conv9_x4v',padding='same',dilation_rate=2)(x4)
x4 = L.Add()([x4,x4w,x4v])
x4 = L.Activation('tanh')(x4)

x = L.Concatenate()([x1,x2,x3,x4])
x = Conv2D(64,(1,1),name='conv9_concat',padding='same')(x)
x = L.Activation('tanh')(x)


# conv10
x1 = Lambda(lambda x : x[:,:,:, 0:16])(x)
x2 = Lambda(lambda x : x[:,:,:,16:32])(x)
x3 = Lambda(lambda x : x[:,:,:,32:48])(x)
x4 = Lambda(lambda x : x[:,:,:,48:64])(x)

x1w= Conv2D(16,(3,3),name='conv10_x1w',padding='same')(x1)
x1v= Conv2D(16,(3,3),name='conv10_x1v',padding='same',dilation_rate=2)(x1)
x1 = L.Add()([x1,x1w,x1v])
x1 = L.Activation('tanh')(x1)

x2w= Conv2D(16,(3,3),name='conv10_x2w',padding='same')(x2)
x2v= Conv2D(16,(3,3),name='conv10_x2v',padding='same',dilation_rate=2)(x2)
x2 = L.Add()([x2,x2w,x2v])
x2 = L.Activation('tanh')(x2)

x3w= Conv2D(16,(3,3),name='conv10_x3w',padding='same')(x3)
x3v= Conv2D(16,(3,3),name='conv10_x3v',padding='same',dilation_rate=2)(x3)
x3 = L.Add()([x3,x3w,x3v])
x3 = L.Activation('tanh')(x3)

x4w= Conv2D(16,(3,3),name='conv10_x4w',padding='same')(x4)
x4v= Conv2D(16,(3,3),name='conv10_x4v',padding='same',dilation_rate=2)(x4)
x4 = L.Add()([x4,x4w,x4v])
x4 = L.Activation('tanh')(x4)

x = L.Concatenate()([x1,x2,x3,x4])
x = Conv2D(64,(1,1),name='conv10_concat',padding='same')(x)
x = L.Activation('tanh')(x)

# mapping
x = Conv2D( 1,(1,1),name='mapping'  ,padding='same')(x)
x = L.Activation('sigmoid')(x)


x = Lambda(lambda x : x[:,:,:,0])(x)
model = Model(inputs=inputs, outputs=x)
opti = optimizers.Adam(lr=learning_rate)
model.compile(loss='binary_crossentropy',
              optimizer=opti,
              metrics=['mae'])





model.load_weights(pretrain_name,by_name=True)
# model=load_model(model_name)


print (model.summary())


# In[ ]:


callback_list=[ModelCheckpoint(model_name,monitor='val_loss',save_best_only=True,mode='min',verbose=1)]

model.fit(x_train, [y_train],
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_val, [y_val]),shuffle='batch',callbacks=callback_list)


# In[ ]:


model = load_model(model_name)
model.save_weights(weights_name)

