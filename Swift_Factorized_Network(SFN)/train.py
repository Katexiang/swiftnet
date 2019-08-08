import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import sfn
import random
import csv
import math
from scipy import misc

#==============INPUT ARGUMENTS==================
flags = tf.app.flags

#Directory arguments
flags.DEFINE_string('dataset_dir', './dataset', 'The dataset directory to find the train, validation and test images.')
flags.DEFINE_string('logdir', './log/swiftnet', 'The log directory to save your checkpoint and event files.')
#Training arguments
flags.DEFINE_integer('num_classes', 19, 'The number of classes to predict.')
flags.DEFINE_integer('batch_size', 11, 'The batch_size for training.')
flags.DEFINE_integer('eval_batch_size', 8, 'The batch size used for validation.')
flags.DEFINE_integer('image_height',768, "The input height of the images.")
flags.DEFINE_integer('image_width', 768, "The input width of the images.")
flags.DEFINE_integer('num_epochs', 200, "The number of epochs to train your model.")
flags.DEFINE_integer('num_epochs_before_decay', 200, 'The number of epochs before decaying your learning rate.')
flags.DEFINE_float('weight_decay', 1e-4, "The weight decay for ENet convolution layers.")
flags.DEFINE_float('learning_rate_decay_factor', 0.6667, 'The learning rate decay factor.')
flags.DEFINE_float('initial_learning_rate', 4e-4, 'The initial learning rate for your training.')
flags.DEFINE_boolean('Start_train',True, "The input height of the images.")

#

FLAGS = flags.FLAGS

Start_train = FLAGS.Start_train
log_name = 'model.ckpt'

num_classes = FLAGS.num_classes
batch_size = FLAGS.batch_size
eval_batch_size = FLAGS.eval_batch_size 
image_height = FLAGS.image_height
image_width = FLAGS.image_width

#Training parameters
initial_learning_rate = FLAGS.initial_learning_rate
num_epochs_before_decay = FLAGS.num_epochs_before_decay
num_epochs =FLAGS.num_epochs
learning_rate_decay_factor = FLAGS.learning_rate_decay_factor
weight_decay = FLAGS.weight_decay
epsilon = 1e-8


#Directories
dataset_dir = FLAGS.dataset_dir
logdir = FLAGS.logdir

#===============PREPARATION FOR TRAINING==================
#Get the images into a list
image_files = sorted([os.path.join(dataset_dir, 'train', file) for file in os.listdir(dataset_dir + "/train") if file.endswith('.png')])
annotation_files = sorted([os.path.join(dataset_dir, "trainannot", file) for file in os.listdir(dataset_dir + "/trainannot") if file.endswith('.png')])
image_val_files = sorted([os.path.join(dataset_dir, 'val', file) for file in os.listdir(dataset_dir + "/val") if file.endswith('.png')])
annotation_val_files = sorted([os.path.join(dataset_dir, "valannot", file) for file in os.listdir(dataset_dir + "/valannot") if file.endswith('.png')])
#保存到excel
csvname=logdir[6:]+'.csv'
with  open(csvname,'a', newline='') as out:
    csv_write = csv.writer(out,dialect='excel')
    a=[str(i) for i in range(num_classes)]
    csv_write.writerow(a)
#Know the number steps to take before decaying the learning rate and batches per epoch
num_batches_per_epoch = math.ceil(len(image_files) / batch_size)
num_steps_per_epoch = num_batches_per_epoch
decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

#=================CLASS WEIGHTS===============================  
class_weights=np.array(	   
	   [40.69042899, 47.6765088 , 12.70029695, 45.20543212, 45.78372173,
       45.82527748, 48.40614895, 42.75593537,  3.36208549, 14.03151966,
        4.9866471 , 39.25440643, 36.51259517, 32.81231979,  6.69824427,
       33.55546509, 18.48781934, 32.97432129, 46.28665742],dtype=np.float32)

def weighted_cross_entropy(onehot_labels, logits, class_weights,annotations_ohe):
    a=tf.reduce_sum(-tf.log(tf.clip_by_value(logits, 1e-10, 1.0))*onehot_labels*class_weights)
    MASK = tf.reduce_sum(1-annotations_ohe[:,:,:,0])#calculation the pixel number of the meaningful classes.
    return a/MASK


#第一次增强采用最大1，然后亮度0.1
def decode(a,b):
    a = tf.read_file(a)
    a=tf.image.decode_png(a, channels=3)
    a=tf.cast(a,dtype=tf.float32)
    b = tf.read_file(b)
    b = tf.image.decode_png(b,channels=1)
    #random scale
    scale = tf.random_uniform([1],minval=0.75,maxval=1.25,dtype=tf.float32)
    hi=tf.floor(scale*1024)
    wi=tf.floor(scale*2048)
    s=tf.concat([hi,wi],0)
    s=tf.cast(s,dtype=tf.int32)
    a=tf.image.resize_images(a, s,method=0,align_corners=True)
    b=tf.image.resize_images(b, s,method=1,align_corners=True)
    b = tf.image.convert_image_dtype(b, dtype=tf.float32)
    #random crop and flip    
    m=tf.concat([a,b],axis=-1)
    m=tf.image.random_crop(m,[image_height,image_width,4])
    m=tf.image.random_flip_left_right(m)
	
    m=tf.split(m,num_or_size_splits=4,axis=-1)
    a=tf.concat([m[0],m[1],m[2]],axis=-1)
    img=tf.image.convert_image_dtype(a/255,dtype=tf.uint8)
    a=a-[123.68,116.779,103.939]
    b=m[3]
    b = tf.image.convert_image_dtype(b, dtype=tf.uint8)
    a.set_shape(shape=(image_height, image_width, 3))
    b.set_shape(shape=(image_height, image_width,1))
    img.set_shape(shape=(image_height, image_width, 3))
    return a,b,img
def decodev(a,b):
    a = tf.read_file(a)
    a=tf.image.decode_png(a, channels=3)
    a=tf.cast(a,dtype=tf.float32)
    b = tf.read_file(b)
    a = a-[123.68,116.779,103.939]
    b = tf.image.decode_png(b,channels=1)
    a.set_shape(shape=(1024, 2048, 3))
    b.set_shape(shape=(1024, 2048,1))
    return a,b
def run():
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)
        #===================TRAINING BRANCH=======================
        #Load the files into one input queue
        images = tf.convert_to_tensor(image_files)
        annotations = tf.convert_to_tensor(annotation_files)
        tdataset = tf.data.Dataset.from_tensor_slices((images,annotations)).map(decode).shuffle(100).batch(batch_size).repeat(num_epochs)
        titerator = tdataset.make_initializable_iterator()
        images,annotations,realimg = titerator.get_next()		
        images_val = tf.convert_to_tensor(image_val_files)
        annotations_val = tf.convert_to_tensor(annotation_val_files)
        vdataset = tf.data.Dataset.from_tensor_slices((images_val,annotations_val)).map(decodev).batch(eval_batch_size).repeat(num_epochs*3)
        viterator = vdataset.make_initializable_iterator()
        images_val,annotations_val = viterator.get_next()				
        #perform one-hot-encoding on the ground truth annotation to get same shape as the logits
        _, probabilities= sfn.swiftnet(images, numclass=num_classes,is_training=True,shape=[image_height,image_width],l2=weight_decay,dropout=0.05,reuse=None)
        annotations = tf.reshape(annotations, shape=[-1, image_height, image_width])
        #loss function
        raw_gt = tf.reshape(annotations, [-1,])
        indices = tf.squeeze(tf.where(tf.greater(raw_gt,0)), 1)
        gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
        gt = gt-1		
        gt_one = tf.one_hot(gt, num_classes, axis=-1)		
        raw_prediction = tf.reshape(probabilities, [-1, num_classes])
        prediction = tf.gather(raw_prediction, indices)
        var=tf.global_variables()		
        var1=[v for v in var if v.name.split('/')[0]=='Resnet18' and v.name.split('/')[-2]!='attention' and  v.name.split('/')[-2]!='attentionbn']	#base_net parameters	
        var2=[v for v in var if v not in var1] #added parameters		
        annotations_ohe = tf.one_hot(annotations, num_classes+1, axis=-1)
        los=weighted_cross_entropy(gt_one, prediction, class_weights,annotations_ohe)
        loss=tf.losses.add_loss(los)
        total_loss = tf.losses.get_total_loss()
        global_step =  tf.train.get_or_create_global_step()
        #Define your  learning rate and optimizer
        lr=tf.train.cosine_decay(
            learning_rate=initial_learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            alpha=2.5e-3)
        optimizer1 =  tf.train.AdamOptimizer(learning_rate=lr/4, epsilon=epsilon)
        optimizer2 =  tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        updates_op = tf.group(*update_ops)
        with tf.control_dependencies([updates_op]):
            grads = tf.gradients(total_loss, var1 + var2)			
            grads1 = grads[:len(var1)]
            grads2 = grads[len(var1):]
            train_op1 = optimizer1.apply_gradients(zip(grads1, var1))
            train_op2 = optimizer2.apply_gradients(zip(grads2, var2),global_step=global_step)
            train_op = tf.group(train_op1, train_op2)
        _, probabilities_val= sfn.swiftnet(images_val, numclass=num_classes,is_training=None,shape=[1024,2048],l2=None,dropout=0,reuse=True)
        raw_gt_v = tf.reshape(tf.reshape(annotations_val, shape=[-1, 1024, 2048]),[-1,])
        indices_v = tf.squeeze(tf.where(tf.greater(raw_gt_v,0)), 1)
        gt_v = tf.cast(tf.gather(raw_gt_v, indices_v), tf.int32)
        gt_v = gt_v-1
        gt_one_v = tf.one_hot(gt_v, num_classes, axis=-1)
        raw_prediction_v = tf.argmax(tf.reshape(probabilities_val, [-1, num_classes]),-1)
        prediction_v = tf.gather(raw_prediction_v, indices_v)
        prediction_ohe_v = tf.one_hot(prediction_v, num_classes, axis=-1)
        and_val=gt_one_v*prediction_ohe_v
        and_sum=tf.reduce_sum(and_val,[0])
        or_val=tf.to_int32((gt_one_v+prediction_ohe_v)>0.5)
        or_sum=tf.reduce_sum(or_val,axis=[0])
        T_sum=tf.reduce_sum(gt_one_v,axis=[0])
        R_sum = tf.reduce_sum(prediction_ohe_v,axis=[0])
        mPrecision=0
        mRecall_rate=0
        mIoU=0	
        #Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
        def train_step(sess, train_op, global_step ,loss=total_loss):
            #Check the time for each sess run
            start_time = time.time()
            _,total_loss, global_step_count= sess.run([train_op,loss, global_step ])
            time_elapsed = time.time() - start_time
            global_step_count=global_step_count+1
            #Run the logging to show some results
            logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)

            return total_loss
        #Now finally create all the summaries you need to monitor and group them into one summary op.
        A = tf.Variable(tf.constant(0.0), dtype=tf.float32)
        a=tf.placeholder(shape=[],dtype=tf.float32)
        Precision=tf.assign(A, a)
        B = tf.Variable(tf.constant(0.0), dtype=tf.float32)
        b=tf.placeholder(shape=[],dtype=tf.float32)
        Recall=tf.assign(B, b)
        C = tf.Variable(tf.constant(0.0), dtype=tf.float32)
        c=tf.placeholder(shape=[],dtype=tf.float32)
        mIOU=tf.assign(C, c)	
        predictions = tf.argmax(probabilities, -1)
        segmentation_output = tf.cast(tf.reshape((predictions+1)*255/num_classes, shape=[-1, image_height, image_width, 1]),tf.uint8)
        segmentation_ground_truth = tf.cast(tf.reshape(tf.cast(annotations, dtype=tf.float32)*255/num_classes, shape=[-1, image_height, image_width, 1]),tf.uint8)		
        tf.summary.scalar('Monitor/Total_Loss', total_loss)
        tf.summary.scalar('Monitor/Precision', Precision)
        tf.summary.scalar('Monitor/Recall_rate', Recall)
        tf.summary.scalar('Monitor/mIoU', mIOU)
        tf.summary.scalar('Monitor/learning_rate', lr)
        tf.summary.image('Images/original_image', realimg, max_outputs=1)
        tf.summary.image('Images/segmentation_output', segmentation_output, max_outputs=1)
        tf.summary.image('Images/segmentation_ground_truth', segmentation_ground_truth, max_outputs=1)
        my_summary_op = tf.summary.merge_all()

        def train_sum(sess, train_op, global_step,sums,loss=total_loss,pre=0,recall=0,iou=0):
            start_time = time.time()
            _,total_loss, global_step_count,ss = sess.run([train_op,loss, global_step,sums ],feed_dict={a:pre,b:recall,c:iou})
            time_elapsed = time.time() - start_time
            global_step_count=global_step_count+1
            logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)

            return total_loss,ss
			
        def eval_step(sess,i ):
            and_eval_batch,or_eval_batch,T_eval_batch,R_eval_batch = sess.run([and_sum,or_sum,T_sum,R_sum])
            #Log some information
            logging.info('STEP: %d ',i)
            return  and_eval_batch,or_eval_batch,T_eval_batch,R_eval_batch
        def eval(num_class,csvname,session,image_val,eval_batch):
            or_=np.zeros((num_class), dtype=np.float32)
            and_=np.zeros((num_class), dtype=np.float32)			
            T_=np.zeros((num_class), dtype=np.float32)			
            R_=np.zeros((num_class), dtype=np.float32)			
            for i in range(math.ceil(len(image_val) / eval_batch)):
                and_eval_batch,or_eval_batch,T_eval_batch,R_eval_batch = eval_step(session,i+1)
                and_=and_+and_eval_batch
                or_=or_+or_eval_batch
                T_=T_+T_eval_batch
                R_=R_+R_eval_batch
            Recall_rate=and_/T_
            Precision=and_/R_
            IoU=and_/or_
            mPrecision=np.mean(Precision)
            mRecall_rate=np.mean(Recall_rate)
            mIoU=np.mean(IoU)
            print("Precision:")
            print(Precision)
            print("Recall rate:")
            print(Recall_rate)
            print("IoU:")
            print(IoU)
            print("mPrecision:")
            print(mPrecision)
            print("mRecall_rate:")
            print(mRecall_rate)
            print("mIoU")
            print(mIoU)
            with open(csvname,'a', newline='') as out:
                csv_write = csv.writer(out,dialect='excel')
                csv_write.writerow(Precision)
                csv_write.writerow(Recall_rate)
                csv_write.writerow(IoU)
            return mPrecision,mPrecision,mIoU
			
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        init = tf.global_variables_initializer()
        saver=tf.train.Saver(var_list=tf.global_variables(),max_to_keep=10)
        with tf.Session(config=config) as sess:
            sess.run(init)
            sess.run([titerator.initializer,viterator.initializer])
            sfn.load_weight(sess,'imgnet_resnet18.npz',var1)#load base_net's parameter.
            step = 0;
            if Start_train is not True:
                #input the checkpoint address,and the step number.
                checkpoint='./log/swiftnet/model.ckpt-37127'
                saver.restore(sess, checkpoint)
                step = 37127
                sess.run(tf.assign(global_step,step))
            summary_writer = tf.summary.FileWriter(logdir, sess.graph)
            final = num_steps_per_epoch * num_epochs
            for i in range(step,final,1):
                if i % num_batches_per_epoch == 0:
                    logging.info('Epoch %s/%s', i/num_batches_per_epoch + 1, num_epochs)
                    learning_rate_value = sess.run([lr])
                    logging.info('Current Learning Rate: %s', learning_rate_value)
                    if i is not step:
                        saver.save(sess, os.path.join(logdir,log_name),global_step=i)					
                        mPrecision,mRecall_rate,mIoU=eval(num_class=num_classes,csvname=csvname,session=sess,image_val=image_val_files,eval_batch=eval_batch_size)                       				
                if i % min(num_steps_per_epoch, 10) == 0:
                    loss,summaries = train_sum(sess, train_op,global_step,sums=my_summary_op,loss=total_loss,pre=mPrecision,recall=mPrecision,iou=mIoU)
                    summary_writer.add_summary(summaries,global_step=i+1)
                else:
                    loss = train_step(sess, train_op, global_step)
            summary_writer.close()					
            eval(num_class=num_classes,csvname=csvname,session=sess,image_val=image_val_files,eval_batch=eval_batch_size)
            logging.info('Final Loss: %s', loss)
            logging.info('Finished training! Saving model to disk now.')
            saver.save(sess,  os.path.join(logdir,log_name), global_step = final)


if __name__ == '__main__':
    run()
