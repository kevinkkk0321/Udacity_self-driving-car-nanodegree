#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # Implement function
    # layer 7 1x1
    layer_7_conv_1x1 = tf.layers.conv2d(
		vgg_layer7_out, num_classes, 1, padding='same',
		kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # layer 7 1x1 deconvolution
    deconv_1 = tf.layers.conv2d_transpose(
		layer_7_conv_1x1, num_classes, 4, strides = (2, 2), padding = 'same',
		kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # layer 4 1x1
    layer_4_conv_1x1 = tf.layers.conv2d(
		vgg_layer4_out, num_classes, 1, padding='same',
		kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # layer 4 skip
    skip_1 = tf.add(deconv_1, layer_4_conv_1x1)
    # Upsampling step. using conv2d transpose function. Kernel is 4 and stride is 2 .
    # Layer size becomes 2x2
    deconv_2 = tf.layers.conv2d_transpose(
		skip_1, num_classes, 4, strides = (2, 2), padding='same',
		kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # layer 3 1x1
    layer_3_conv_1x1 = tf.layers.conv2d(
		vgg_layer3_out, num_classes, 1, padding='same',
		kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # layer 3 skip
    skip_2 = tf.add(deconv_2, layer_3_conv_1x1)
    # final Upsampling
    deconv_3 = tf.layers.conv2d_transpose(
		skip_2, num_classes, 16, strides=(8, 8), padding='same',
		kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return deconv_3
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)



def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # Implement function
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        sum_loss = 0
        num_batch = 0
        keep_prob_value = 0.5
        learning_rate_value = 0.0001

        print('epoch : %d'  % epoch)
        for image, label in get_batches_fn(batch_size):
<<<<<<< HEAD
<<<<<<< HEAD
            _, loss = sess.run([train_op, cross_entropy_loss],
=======
            _, batch_cost = sess.run([train_op, cross_entropy_loss],
>>>>>>> 21b89813ed9322ff1549e5b385e94fe683a03ae5
=======
            _, batch_cost = sess.run([train_op, cross_entropy_loss],
>>>>>>> 21b89813ed9322ff1549e5b385e94fe683a03ae5
                    feed_dict={
                    input_image: image,
                    correct_label: label,
                    keep_prob: 0.5,
                    learning_rate: learning_rate_value,
                    })
            sum_loss += loss
            num_batch += 1
            if num_batch % 10 == 0:
<<<<<<< HEAD
<<<<<<< HEAD
                print('loss %.9f' % loss)

        avg_loss = sum_loss / num_batch
        print('Epoch: %04d, average loss=%.9f' % ((epoch+1), avg_loss))
        #print('Saver: %s' % saver)
        #if saver is not None:
        #    saver.save(sess, './saved_model/checkpoint', global_step=(epoch+1))
=======
=======
>>>>>>> 21b89813ed9322ff1549e5b385e94fe683a03ae5
                print('loss %.9f' % batch_cost)

        avg_loss = sum_loss / num_batch
        print('Epoch: %04d, average loss=%.9f' % ((epoch+1), avg_loss))
        print('Saver: %s' % saver)
        if saver is not None:
            saver.save(sess, './saved_model/checkpoint', global_step=(epoch+1))
<<<<<<< HEAD
>>>>>>> 21b89813ed9322ff1549e5b385e94fe683a03ae5
=======
>>>>>>> 21b89813ed9322ff1549e5b385e94fe683a03ae5


tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    export_dir = './saved_model'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # Save model and freeze it later
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)


    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)
        # TODO: Train NN using the train_nn function
        #initialize all variables
        epochs = 30
        batch_size = 8
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess.run(init)

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                     correct_label, keep_prob, learning_rate)


        print('Saving model and variables to %s' % export_dir)
        builder.add_meta_graph_and_variables(sess, ['carnd'])
        builder.save()


        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video



if __name__ == '__main__':
    run()
