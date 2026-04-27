import os
import sys
import pickle
import random  # Make sure to import random if you're shuffling lists

import numpy as np
import tensorflow as tf
import prepare.data_transformation as dt
import matplotlib.pyplot as plt  # Added import for plotting

################################################################### Function
##### Activation
def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

##### Round
def threshold(x, val=0.5):
    x = tf.clip_by_value(x, 0.5, 0.5001) - 0.5
    x = tf.minimum(x * 10000, 1)
    return x

##### Neuron Networks
def decoder(z, phase_train=True, reuse=False):
    strides = [1, 2, 2, 2, 1]
    with tf.variable_scope("gen", reuse=reuse):
        batch_size = tf.shape(z)[0]  # Dynamic batch size
        z = tf.reshape(z, [batch_size, 1, 1, 1, z_size])
        g_1 = tf.nn.conv3d_transpose(z, weights['wg1'],
                                     output_shape=tf.stack([batch_size, 4, 4, 4, 64]),
                                     strides=[1, 1, 1, 1, 1], padding="VALID")
        g_1 = lrelu(g_1)

        g_2 = tf.nn.conv3d_transpose(g_1, weights['wg2'],
                                     output_shape=tf.stack([batch_size, 8, 8, 8, 64]),
                                     strides=strides, padding="SAME")
        g_2 = lrelu(g_2)

        g_3 = tf.nn.conv3d_transpose(g_2, weights['wg3'],
                                     output_shape=tf.stack([batch_size, 16, 16, 16, 64]),
                                     strides=strides, padding="SAME")
        g_3 = lrelu(g_3)

        g_4 = tf.nn.conv3d_transpose(g_3, weights['wg4'],
                                     output_shape=tf.stack([batch_size, 32, 32, 32, 64]),
                                     strides=strides, padding="SAME")
        g_4 = lrelu(g_4)

        g_5 = tf.nn.conv3d_transpose(g_4, weights['wg5'],
                                     output_shape=tf.stack([batch_size, 64, 64, 64, 1]),
                                     strides=strides, padding="SAME")
        g_5 = tf.nn.sigmoid(g_5)

        return g_5

def encoder(inputs, phase_train=True, reuse=False):
    leak_value = 0.2
    strides = [1, 2, 2, 2, 1]
    with tf.variable_scope("enc", reuse=reuse):
        d_1 = tf.nn.conv3d(inputs, weights['wae1'], strides=[1, 2, 2, 2, 1], padding="SAME")
        d_1 = lrelu(d_1, leak_value)

        d_2 = tf.nn.conv3d(d_1, weights['wae2'], strides=strides, padding="SAME")
        d_2 = lrelu(d_2, leak_value)

        d_3 = tf.nn.conv3d(d_2, weights['wae3'], strides=strides, padding="SAME")
        d_3 = lrelu(d_3, leak_value)

        d_4 = tf.nn.conv3d(d_3, weights['wae4'], strides=[1, 2, 2, 2, 1], padding="SAME")
        d_4 = lrelu(d_4, leak_value)

        d_5 = tf.nn.conv3d(d_4, weights['wae5'], strides=[1, 1, 1, 1, 1], padding="VALID")
        d_5 = tf.nn.tanh(d_5)

        return d_5

##### Weight Initialization
weights = {}

def initialiseWeights():
    global weights
    xavier_init = tf.contrib.layers.xavier_initializer()

    weights['wg1'] = tf.get_variable("wg1", shape=[4, 4, 4, 64, z_size], initializer=xavier_init)
    weights['wg2'] = tf.get_variable("wg2", shape=[4, 4, 4, 64, 64], initializer=xavier_init)
    weights['wg3'] = tf.get_variable("wg3", shape=[4, 4, 4, 64, 64], initializer=xavier_init)
    weights['wg4'] = tf.get_variable("wg4", shape=[4, 4, 4, 64, 64], initializer=xavier_init)
    weights['wg5'] = tf.get_variable("wg5", shape=[4, 4, 4, 1, 64], initializer=xavier_init)

    weights['wae1'] = tf.get_variable("wae1", shape=[4, 4, 4, 1, 64], initializer=xavier_init)
    weights['wae2'] = tf.get_variable("wae2", shape=[4, 4, 4, 64, 64], initializer=xavier_init)
    weights['wae3'] = tf.get_variable("wae3", shape=[4, 4, 4, 64, 64], initializer=xavier_init)
    weights['wae4'] = tf.get_variable("wae4", shape=[4, 4, 4, 64, 64], initializer=xavier_init)
    weights['wae5'] = tf.get_variable("wae5", shape=[4, 4, 4, 64, z_size], initializer=xavier_init)

    return weights

########################################################################### Training
##### Parameters
batch_size = 16  # Adjusted as per your requirement
z_size = 200
reg_l2 = 1e-4  # Regularization coefficient
ae_lr = 0.0003
n_ae_epochs = 101
number_of_different_element = 2
patience = 10  # For early stopping

def sites_autocoder(sites_graph_path='./test_sites/',
                    encoded_graph_path='./test_encoded_sites/',
                    model_path='./sites_model/'):
    tf.reset_default_graph()
    if not os.path.exists(encoded_graph_path):
        os.makedirs(encoded_graph_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    ##### Initialize Weights
    weights = initialiseWeights()
    x_vector = tf.placeholder(shape=[None, 64, 64, 64, 1],
                              dtype=tf.float32)  # Batch size is dynamic
    z_vector = tf.placeholder(shape=[None, 1, 1, 1, z_size], dtype=tf.float32)

    # Weights for autoencoder pretraining
    with tf.variable_scope('encoders') as scope1:
        encoded = encoder(x_vector, phase_train=True, reuse=False)
        scope1.reuse_variables()
        encoded2 = encoder(x_vector, phase_train=False, reuse=True)

    with tf.variable_scope('gen_from_dec') as scope2:
        decoded = decoder(encoded, phase_train=True, reuse=False)
        scope2.reuse_variables()
        decoded_test = decoder(encoded2, phase_train=False, reuse=True)

    # Round decoder output
    decoded = threshold(decoded)
    decoded_test = threshold(decoded_test)

    # Compute MSE Loss and L2 Loss
    mse_loss = tf.reduce_mean(tf.pow(x_vector - decoded, 2))
    mse_loss2 = tf.reduce_mean(tf.pow(x_vector - decoded_test, 2))
    para_ae = [var for var in tf.trainable_variables() if any(x in var.name for x in ['wae', 'wg'])]
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in para_ae])
    ae_loss = mse_loss + reg_l2 * l2_loss
    optimizer_ae = tf.train.AdamOptimizer(learning_rate=ae_lr, name="Adam_AE").minimize(ae_loss, var_list=para_ae)

    saver = tf.train.Saver()

    # Initialize lists to store losses
    training_losses = []
    test_losses = []

    # Early stopping variables
    best_epoch = 0
    best_epoch_loss = float('inf')
    patience_counter = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Prepare data
        test_size, test_name_list, train_name_list = dt.train_test_split(path=sites_graph_path, split_ratio=0.1)
        min_mse_test = float('inf')

        for epoch in range(n_ae_epochs):
            # Shuffle training data at each epoch
            random.shuffle(train_name_list)

            # Create mini-batches
            num_batches = len(train_name_list) // batch_size
            mse_tr = 0
            for batch_idx in range(num_batches):
                batch_names = train_name_list[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                inputs_batch = []
                for name in batch_names:
                    input_path = os.path.join(sites_graph_path, name + '.npy')
                    data = np.load(input_path)
                    # Assuming data shape is (64, 64, 64, number_of_different_element)
                    for i in range(number_of_different_element):
                        element_data = data[:, :, :, i].reshape(64, 64, 64, 1)
                        inputs_batch.append(element_data)
                inputs_batch = np.array(inputs_batch)
                # Reshape to [batch_size * number_of_elements, 64, 64, 64, 1]
                inputs_batch = inputs_batch.reshape(-1, 64, 64, 64, 1)

                # Run optimization
                mse_l, _ = sess.run([mse_loss, optimizer_ae], feed_dict={x_vector: inputs_batch})
                mse_tr += mse_l

            avg_mse_tr = mse_tr / (num_batches * number_of_different_element)

            # Validation/Test loss
            num_test_batches = len(test_name_list) // batch_size
            mse_test = 0
            for batch_idx in range(num_test_batches):
                batch_names = test_name_list[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                test_inputs_batch = []
                for name in batch_names:
                    test_input_path = os.path.join(sites_graph_path, name + '.npy')
                    data = np.load(test_input_path)
                    for i in range(number_of_different_element):
                        element_data = data[:, :, :, i].reshape(64, 64, 64, 1)
                        test_inputs_batch.append(element_data)
                test_inputs_batch = np.array(test_inputs_batch)
                test_inputs_batch = test_inputs_batch.reshape(-1, 64, 64, 64, 1)

                # Compute test loss
                mse_t = sess.run(mse_loss2, feed_dict={x_vector: test_inputs_batch})
                mse_test += mse_t

            avg_mse_test = mse_test / (num_test_batches * number_of_different_element)

            print(f"Epoch {epoch}: Training Loss = {avg_mse_tr:.6f}, Test Loss = {avg_mse_test:.6f}")

            # Append losses to the lists
            training_losses.append(avg_mse_tr)
            test_losses.append(avg_mse_test)

            # Early stopping and saving the best model
            if avg_mse_test < best_epoch_loss and avg_mse_test < 5e-5:
                best_epoch_loss = avg_mse_test
                best_epoch = epoch
                patience_counter = 0
                saver.save(sess, save_path=os.path.join(model_path, 'sites.ckpt'))
                total_name_list = test_name_list + train_name_list
                for name in total_name_list:
                    savefilename = os.path.join(encoded_graph_path, name + '.npy')
                    encoded_sites = np.zeros([z_size, number_of_different_element])
                    for i in range(number_of_different_element):
                        element_input = np.load(os.path.join(sites_graph_path, name + '.npy'))[:, :, :, i].reshape(1,
                                                                                                                   64,
                                                                                                                   64,
                                                                                                                   64,
                                                                                                                   1)
                        encoded_sites[:, i] = sess.run(encoded2, feed_dict={x_vector: element_input}).reshape(z_size)
                    np.save(savefilename, encoded_sites)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Plotting the learning curve after training
        epochs_range = range(1, len(training_losses) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, training_losses, 'k-', label='Training Loss')  # Black line
        plt.plot(epochs_range, test_losses, 'r-', label='Test Loss')  # Red line
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Learning Curve of Sites Autoencoder')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot to a file
        plot_path = os.path.join(model_path, 'sites_learning_curve.png')
        plt.savefig(plot_path)
        print(f'Learning curve saved to {plot_path}')

def sites_restorer(generated_2d_path='./generated_2d_graph/',
                   generated_decoded_path='./generated_decoded_sites/',
                   model_path='./sites_model/'):
    tf.reset_default_graph()
    if not os.path.exists(generated_decoded_path):
        os.makedirs(generated_decoded_path)

    ##### Initialize Weights
    weights = initialiseWeights()
    x_vector = tf.placeholder(shape=[None, 64, 64, 64, 1], dtype=tf.float32)  # Adjusted batch size
    z_vector = tf.placeholder(shape=[None, 1, 1, 1, z_size], dtype=tf.float32)

    # Weights for autoencoder pretraining
    with tf.variable_scope('encoders') as scope1:
        encoded = encoder(x_vector, phase_train=True, reuse=False)
        scope1.reuse_variables()
        encoded2 = encoder(x_vector, phase_train=False, reuse=True)

    with tf.variable_scope('gen_from_dec') as scope2:
        decoded = decoder(encoded, phase_train=True, reuse=False)
        scope2.reuse_variables()
        decoded_test = decoder(encoded2, phase_train=False, reuse=True)

    # Round decoder output
    decoded = threshold(decoded)
    decoded_test = threshold(decoded_test)

    # Compute MSE Loss and L2 Loss
    mse_loss = tf.reduce_mean(tf.pow(x_vector - decoded, number_of_different_element))
    mse_loss2 = tf.reduce_mean(tf.pow(x_vector - decoded_test, number_of_different_element))
    para_ae = [var for var in tf.trainable_variables() if any(x in var.name for x in ['wae', 'wg'])]
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in para_ae])
    ae_loss = mse_loss + reg_l2 * l2_loss
    optimizer_ae = tf.train.AdamOptimizer(learning_rate=ae_lr, name="Adam_AE").minimize(ae_loss, var_list=para_ae)

    restore_saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore the trained model
        restore_saver.restore(sess, os.path.join(model_path, 'sites.ckpt'))

        total_name_list = dt.train_test_split(path=generated_2d_path, split_ratio=0.1)[2] + \
                          dt.train_test_split(path=generated_2d_path, split_ratio=0.1)[1]
        for name in total_name_list:
            savefilename = os.path.join(generated_decoded_path, name + '.npy')
            decoded_sites = np.zeros([64, 64, 64, number_of_different_element])
            ge = np.load(os.path.join(generated_2d_path, name + '.npy'))
            for i in range(number_of_different_element):
                # Assuming 'ge' has at least (1 + number_of_different_element) rows
                ge_element = ge[i + 1, :].reshape(1, 1, 1, 1, z_size)
                decoded_sites[:, :, :, i] = sess.run(decoded_test, feed_dict={encoded2: ge_element}).reshape(64, 64, 64)
            np.save(savefilename, decoded_sites)
