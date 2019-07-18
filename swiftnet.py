import tensorflow as tf
import math
import numpy as np





def conv(inputs,filters,kernel_size,strides=(1, 1),padding='SAME',dilation_rate=(1, 1),activation=tf.nn.relu,use_bias=None,regularizer=None,name=None,reuse=None):
    out=tf.layers.conv2d(
    inputs,
    filters=filters,
    kernel_size=kernel_size,
    strides=strides,
    padding=padding,
    dilation_rate=dilation_rate,
    activation=activation,
    use_bias=use_bias,
    kernel_regularizer=regularizer,
    bias_initializer=tf.zeros_initializer(),
    kernel_initializer= tf.random_normal_initializer(stddev=0.1),
    name=name,
    reuse=reuse)
    return out


# In[3]:


def batch(inputs,training=True,reuse=None,momentum=0.9,name='n'):
    out=tf.layers.batch_normalization(inputs,training=training,reuse=reuse,momentum=momentum,name=name)
    return out


# In[4]:


def branch1(x,numOut,l2,stride=1,is_training=True,momentum=0.9,reuse=None):
    reg = None if l2 is None else tf.contrib.layers.l2_regularizer(scale=l2)
    with tf.variable_scope("conv1"):
        y = conv(x, numOut, kernel_size=[3, 3],activation=None,strides=(stride,stride),name='conv',regularizer=reg,reuse=reuse)
        y = tf.nn.relu(batch(y,training=is_training,reuse=reuse,momentum=momentum,name='bn'))
    with tf.variable_scope("conv2"):
        y = conv(y, numOut, kernel_size=[3, 3],activation=None,regularizer=reg,name='conv',reuse=reuse)
        y = batch(y,training=is_training,reuse=reuse,momentum=momentum,name='bn')
    return y


# In[5]:


def branch2(x,numOut,l2,stride=1,is_training=True,momentum=0.9,reuse=None):
    reg = None if l2 is None else tf.contrib.layers.l2_regularizer(scale=l2)
    with tf.variable_scope("convshortcut"):
        y = conv(x, numOut, kernel_size=[1, 1],activation=None,strides=(stride,stride),name='conv',regularizer=reg,reuse=reuse)
        y = batch(y,training=is_training,reuse=reuse,momentum=momentum,name='bn')
        return y


# In[6]:


def residual(x,numOut,l2,stride=1,is_training=True,reuse=None,momentum=0.9,branch=False,name='res'):
    with tf.variable_scope(name):
        block = branch1(x,numOut,l2,stride=stride,is_training=is_training,momentum=momentum,reuse=reuse)
        if x.get_shape().as_list()[3] != numOut or branch:
            skip = branch2(x, numOut,l2,stride=stride,is_training=is_training,momentum=momentum,reuse=reuse)
            return tf.nn.relu(block+skip),block+skip
        else:
            return  tf.nn.relu(x+block),x+block


# In[7]:


def resnet18(x, is_training,l2=None,dropout=0.05,reuse=None,momentum=0.9,name='Resnet18'):
    feature=[]
    with tf.variable_scope(name):
        reg = None if l2 is None else tf.contrib.layers.l2_regularizer(scale=l2/4)
        y=conv(x, 64, kernel_size=[7, 7],activation=None,strides=2,name='conv0',regularizer=reg,reuse=reuse)
        y=tf.nn.relu(batch(y,training=is_training,reuse=reuse,momentum=momentum,name='conv0/bn'))
        y=tf.nn.max_pool(y,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool1')
        with tf.variable_scope('group0'):
            res2a,t=residual(y,64,l2,branch=True,reuse=reuse,is_training=is_training,name='block0')
            res2b,t=residual(res2a,64,l2,reuse=reuse,is_training=is_training,name='block1')
            feature.append(t)
        with tf.variable_scope('group1'):
            res3a,t=residual(res2b,128,l2,stride=2,reuse=reuse,is_training=is_training,name='block0')
            res3b,t=residual(res3a,128,l2,reuse=reuse,is_training=is_training,name='block1')
            feature.append(t)
        with tf.variable_scope('group2'):
            res4a,t=residual(res3b,256,l2,stride=2,reuse=reuse,is_training=is_training,name='block0')
            res4b,t=residual(res4a,256,l2,reuse=reuse,is_training=is_training,name='block1')
            feature.append(t)
        with tf.variable_scope('group3'):
            res5a,t=residual(res4b,512,l2,stride=2,reuse=reuse,is_training=is_training,name='block0')
            res5b,t=residual(res5a,512,l2,reuse=reuse,is_training=is_training,name='block1')
            feature.append(t)
        #pool5=tf.reduce_mean(res5b, [1, 2],keepdims=True)
        #dropout = tf.layers.dropout(pool5,rate=dropout,training=is_training)
        #y=conv(dropout, 1000, kernel_size=[1, 1],activation=None,name='class',use_bias=True,regularizer=reg,reuse=reuse)
        #y=conv(y, 512, kernel_size=[1, 1],activation=None,name='attention',use_bias=None,regularizer=reg,reuse=reuse)
        #y=tf.nn.relu(batch(y,training=is_training,reuse=reuse,momentum=momentum,name='bn'))		
        #y=res5b*y
        #feature.append(y)
    return y,feature




def SpatialPyramidPooling(x, is_training,shape=[512,512],grids=(8, 4, 2),l2=None,reuse=None,momentum=0.9,name='spp'):
    levels=[]
    height=math.ceil(shape[0]/32)
    weight=math.ceil(shape[1]/32)
    with tf.variable_scope(name):
        reg = None if l2 is None else tf.contrib.layers.l2_regularizer(scale=l2)
        x=tf.nn.relu(batch(x,training=is_training,reuse=reuse,momentum=momentum,name='bn0'))
        x=conv(x, 128, kernel_size=1,activation=None,name='conv0',regularizer=reg,reuse=reuse)
        levels.append(x)
        for i in range(len(grids)):
            h=math.floor(height/grids[i])
            w=math.floor(weight/grids[i])
            kh=height-(grids[i]-1) * h
            kw=weight-(grids[i]-1) * w
            y=tf.nn.avg_pool(x,[1,kh,kw,1],[1,h,w,1],padding='VALID')
            y=tf.nn.relu(batch(y,training=is_training,reuse=reuse,momentum=momentum,name='bn'+str(i+1)))
            y=conv(y, 42, kernel_size=1,activation=None,name='conv'+str(i+1),regularizer=reg,reuse=reuse)
            y=tf.image.resize_images(y, [height,weight],method=0,align_corners=True)
            levels.append(y)
        final=tf.concat(levels,-1)
        final=tf.nn.relu(batch(final,training=is_training,reuse=reuse,momentum=momentum,name='blendbn'))
        final=conv(final, 128, kernel_size=1,activation=None,name='blendconv',regularizer=reg,reuse=reuse)
    return final



def upsample(x,skip,is_training,shape=[512,512],stage=0,l2=None,reuse=None,momentum=0.9,name='up0'):
    height=math.ceil(shape[0]/math.pow(2,5-stage))
    weight=math.ceil(shape[1]/math.pow(2,5-stage))
    with tf.variable_scope(name):
        reg = None if l2 is None else tf.contrib.layers.l2_regularizer(scale=l2)
        skip=tf.nn.relu(batch(skip,training=is_training,reuse=reuse,momentum=momentum,name='skipbn'))
        skip=conv(skip, 128, kernel_size=1,activation=None,name='skipconv',regularizer=reg,reuse=reuse)
        x=tf.image.resize_images(x, [height,weight],method=0,align_corners=True)
        x=x+skip
        x=tf.nn.relu(batch(x,training=is_training,reuse=reuse,momentum=momentum,name='blendbn'))
        x=conv(x, 128, kernel_size=3,activation=None,name='blendconv',regularizer=reg,reuse=reuse)
        return x



def swiftnet(x, numclass,is_training,shape,l2=None,dropout=0.05,reuse=None,momentum=0.9):
    xclass,feature=resnet18(x, is_training,l2,dropout=dropout,reuse=reuse,momentum=momentum,name='Resnet18')
    x=SpatialPyramidPooling(feature[-1], is_training,shape=shape,grids=(8, 4, 2),l2=l2,reuse=reuse,momentum=momentum,name='spp')
    x=upsample(x,feature[-2],is_training,shape=shape,stage=1,l2=l2,reuse=reuse,momentum=momentum,name='up1')
    x=upsample(x,feature[-3],is_training,shape=shape,stage=2,l2=l2,reuse=reuse,momentum=momentum,name='up2')
    x=upsample(x,feature[-4],is_training,shape=shape,stage=3,l2=l2,reuse=reuse,momentum=momentum,name='up3')
    with tf.variable_scope('class'):
        reg = None if l2 is None else tf.contrib.layers.l2_regularizer(scale=l2)
        x=tf.nn.relu(batch(x,training=is_training,reuse=reuse,momentum=momentum,name='classbn'))
        x=conv(x, numclass, kernel_size=3,activation=None,name='classconv',regularizer=reg,reuse=reuse)
        x=tf.image.resize_images(x, [shape[0],shape[1]],method=0,align_corners=True)
        final=tf.nn.softmax(x, name='logits_to_softmax')
    return x,final







def load_weight(sess,resnet50_path,varss):
    param = dict(np.load(resnet50_path))
    for v in varss:
        nameEnd = v.name.split('/')[-1]
        if nameEnd == "moving_mean:0":
            name =  v.name[9:-13]+"mean/EMA"
        elif nameEnd == "moving_variance:0":
            name = v.name[9:-17]+"variance/EMA"
        elif nameEnd =='kernel:0':
            if v.name.split('/')[1]=='conv0':
                name='conv0/W'
                b=np.expand_dims(param[name][:,:,0,:],2)
                g=np.expand_dims(param[name][:,:,1,:],2)
                r=np.expand_dims(param[name][:,:,2,:],2)
                param[name]=np.concatenate([r,g,b],2)
            elif v.name.split('/')[1]=='class':
                name='linear/W'
            else:
                name=v.name[9:-13]+'W'
        elif nameEnd=='gamma:0':
            name=v.name[9:-2]
        elif nameEnd=='beta:0':
            name=v.name[9:-2]
        else:
            name='linear/b'
        sess.run(v.assign(param[name]))
        print("Copy weights: " + name + "---->"+ v.name)


