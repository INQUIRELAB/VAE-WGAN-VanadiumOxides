import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import prepare.data_transformation as dt
import matplotlib.pyplot as plt

# Import Keras backend
import tensorflow.keras.backend as K

###################################################################function
#####activation
def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

##### Custom 3D upsampling function
def upsample_3d(x, scale):
    x = K.repeat_elements(x, rep=scale[0], axis=1)  # Depth axis
    x = K.repeat_elements(x, rep=scale[1], axis=2)  # Height axis
    x = K.repeat_elements(x, rep=scale[2], axis=3)  # Width axis
    return x

#####neuron networks
def decoder(z, is_training, reuse=False):
    strides = [1, 2, 2, 2, 1]
    with tf.variable_scope("decoder", reuse=reuse):
        z = tf.reshape(z, (-1, 1, 1, 1, z_size))  # Use -1 for batch size
        g_1 = tf.nn.conv3d_transpose(z, weights['wd1'], output_shape=[tf.shape(z)[0], 4, 4, 4, 64],
                                     strides=[1, 1, 1, 1, 1], padding="VALID")
        g_1 = tf.layers.batch_normalization(g_1, training=is_training)
        g_1 = lrelu(g_1)
        g_1 = tf.layers.dropout(g_1, rate=0.3, training=is_training)

        # Upsample g_1 for residual connection
        g_1_upsampled = upsample_3d(g_1, scale=(2, 2, 2))

        g_2 = tf.nn.conv3d_transpose(g_1, weights['wd2'], output_shape=[tf.shape(z)[0], 8, 8, 8, 64],
                                     strides=strides, padding="SAME")
        g_2 = tf.layers.batch_normalization(g_2, training=is_training)
        g_2 = lrelu(g_2)
        g_2 = tf.layers.dropout(g_2, rate=0.3, training=is_training)

        # Add residual connection
        g_2 += g_1_upsampled

        # Upsample g_2 for next residual connection
        g_2_upsampled = upsample_3d(g_2, scale=(2, 2, 2))

        g_3 = tf.nn.conv3d_transpose(g_2, weights['wd3'], output_shape=[tf.shape(z)[0], 16, 16, 16, 64],
                                     strides=strides, padding="SAME")
        g_3 = tf.layers.batch_normalization(g_3, training=is_training)
        g_3 = lrelu(g_3)
        g_3 = tf.layers.dropout(g_3, rate=0.3, training=is_training)

        # Add residual connection
        g_3 += g_2_upsampled

        # Final layer
        g_4 = tf.nn.conv3d_transpose(g_3, weights['wd4'], output_shape=[tf.shape(z)[0], 32, 32, 32, 1],
                                     strides=[1, 2, 2, 2, 1], padding="SAME")
        g_4 = tf.nn.sigmoid(g_4)

        return g_4

def encoder(inputs, is_training, reuse=False):
    leak_value = 0.2
    strides = [1, 2, 2, 2, 1]
    with tf.variable_scope("encoder", reuse=reuse):
        e_1 = tf.nn.conv3d(inputs, weights['we1'], strides=strides, padding="SAME")
        e_1 = tf.layers.batch_normalization(e_1, training=is_training)
        e_1 = lrelu(e_1, leak_value)
        e_1 = tf.layers.dropout(e_1, rate=0.3, training=is_training)

        e_2 = tf.nn.conv3d(e_1, weights['we2'], strides=strides, padding="SAME")
        e_2 = tf.layers.batch_normalization(e_2, training=is_training)
        e_2 = lrelu(e_2, leak_value)
        e_2 = tf.layers.dropout(e_2, rate=0.3, training=is_training)

        e_3 = tf.nn.conv3d(e_2, weights['we3'], strides=strides, padding="SAME")
        e_3 = tf.layers.batch_normalization(e_3, training=is_training)
        e_3 = lrelu(e_3, leak_value)
        e_3 = tf.layers.dropout(e_3, rate=0.3, training=is_training)

        e_4 = tf.nn.conv3d(e_3, weights['we4'], strides=[1, 1, 1, 1, 1], padding="VALID")
        e_4 = tf.layers.batch_normalization(e_4, training=is_training)
        e_4 = lrelu(e_4, leak_value)
        e_4 = tf.layers.dropout(e_4, rate=0.3, training=is_training)

        # Flatten and output mean and log variance for reparameterization trick
        e_4_flat = tf.reshape(e_4, [-1, z_size])
        z_mean = tf.layers.dense(e_4_flat, z_size)
        z_logvar = tf.layers.dense(e_4_flat, z_size)

        return z_mean, z_logvar

#####weight
weights = {}

def initialiseWeights():
    global weights
    xavier_init = tf.contrib.layers.xavier_initializer()

    weights['wd1'] = tf.get_variable("wd1", shape=[4, 4, 4, 64, z_size], initializer=xavier_init)
    weights['wd2'] = tf.get_variable("wd2", shape=[4, 4, 4, 64, 64], initializer=xavier_init)
    weights['wd3'] = tf.get_variable("wd3", shape=[4, 4, 4, 64, 64], initializer=xavier_init)
    weights['wd4'] = tf.get_variable("wd4", shape=[4, 4, 4, 1, 64], initializer=xavier_init)

    weights['we1'] = tf.get_variable("we1", shape=[4, 4, 4, 1, 64], initializer=xavier_init)
    weights['we2'] = tf.get_variable("we2", shape=[4, 4, 4, 64, 64], initializer=xavier_init)
    weights['we3'] = tf.get_variable("we3", shape=[4, 4, 4, 64, 64], initializer=xavier_init)
    weights['we4'] = tf.get_variable("we4", shape=[4, 4, 4, 64, z_size], initializer=xavier_init)

    return weights

###########################################################################training
#####parameters
batch_size = 32
z_size = 25
reg_l2 = 1e-6
initial_ae_lr = 0.0003
n_ae_epochs = 201
beta = 1.0  # Beta for Beta-VAE, can be adjusted
kl_annealing_epochs = 100  # Number of epochs to anneal KL weight

def lattice_vae(lattice_graph_path='./test_lattice/', encoded_graph_path='./test_encoded_lattice/',
                model_path='./test_model/'):
    tf.reset_default_graph()
    if not os.path.exists(encoded_graph_path):
        os.makedirs(encoded_graph_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    #####train_function
    weights = initialiseWeights()
    x_vector = tf.placeholder(shape=[None, 32, 32, 32, 1], dtype=tf.float32)
    is_training = tf.placeholder(tf.bool, name='is_training')
    kl_weight = tf.placeholder(tf.float32, name='kl_weight')

    # Encoder outputs
    z_mean, z_logvar = encoder(x_vector, is_training=is_training, reuse=False)

    # Reparameterization trick
    eps = tf.random_normal(shape=tf.shape(z_mean))
    z = z_mean + tf.exp(z_logvar / 2) * eps

    # Decoder outputs
    x_recon = decoder(z, is_training=is_training, reuse=False)

    # Reconstruction loss (MSE)
    recon_loss = tf.reduce_mean(tf.pow(x_vector - x_recon, 2))

    # KL divergence loss
    kl_loss = -0.5 * tf.reduce_mean(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))

    # Total loss with beta and KL annealing
    total_loss = recon_loss + kl_weight * beta * kl_loss

    # Regularization losses
    para_vae = [var for var in tf.trainable_variables() if any(x in var.name for x in ['we', 'wd'])]
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in para_vae])
    total_loss += reg_l2 * l2_loss

    # Learning rate scheduler
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(initial_ae_lr, global_step,
                                               decay_steps=1000, decay_rate=0.96, staircase=True)

    # Ensure batch normalization moving averages are updated
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss, global_step=global_step)

    saver = tf.train.Saver()

    # Initialize lists to store losses
    training_losses = []
    validation_losses = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test_size, test_name_list, train_name_list = dt.train_test_split(path=lattice_graph_path, split_ratio=0.1)
        min_val_loss = np.inf

        num_train_samples = len(train_name_list)
        num_train_batches = (num_train_samples + batch_size - 1) // batch_size

        num_test_samples = len(test_name_list)
        num_test_batches = (num_test_samples + batch_size - 1) // batch_size

        for epoch in range(n_ae_epochs):
            # KL annealing
            current_kl_weight = (epoch / kl_annealing_epochs) if epoch < kl_annealing_epochs else 1.0

            train_loss = 0
            val_loss = 0

            # Shuffle the training data
            np.random.shuffle(train_name_list)

            for batch_idx in range(num_train_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_train_samples)
                batch_names = train_name_list[start_idx:end_idx]

                inputs_batch = []
                for name in batch_names:
                    input_path = os.path.join(lattice_graph_path, name + '.npy')
                    inputs_batch.append(np.load(input_path).reshape(32, 32, 32, 1))

                inputs_batch = np.stack(inputs_batch)
                feed_dict = {
                    x_vector: inputs_batch,
                    is_training: True,
                    kl_weight: current_kl_weight
                }
                _, loss_value = sess.run([optimizer, total_loss], feed_dict=feed_dict)
                train_loss += loss_value * len(batch_names)

            # Compute average training loss
            avg_train_loss = train_loss / num_train_samples

            # Validation
            for batch_idx in range(num_test_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_test_samples)
                batch_names = test_name_list[start_idx:end_idx]

                inputs_batch = []
                for name in batch_names:
                    input_path = os.path.join(lattice_graph_path, name + '.npy')
                    inputs_batch.append(np.load(input_path).reshape(32, 32, 32, 1))

                inputs_batch = np.stack(inputs_batch)
                feed_dict = {
                    x_vector: inputs_batch,
                    is_training: False,
                    kl_weight: 1.0  # Use full KL weight during validation
                }
                loss_value = sess.run(total_loss, feed_dict=feed_dict)
                val_loss += loss_value * len(batch_names)

            # Compute average validation loss
            avg_val_loss = val_loss / num_test_samples

            print(f'Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, KL Weight: {current_kl_weight:.4f}')

            # Append losses to the lists
            training_losses.append(avg_train_loss)
            validation_losses.append(avg_val_loss)

            # Early stopping and checkpointing
            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                saver.save(sess, save_path=os.path.join(model_path, 'lattice_vae.ckpt'))
                print(f'Model saved at epoch {epoch}')

        # Plotting the learning curve after training
        epochs_list = range(n_ae_epochs)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_list, training_losses, 'k-', label='Training Loss')  # Black line
        plt.plot(epochs_list, validation_losses, 'r-', label='Validation Loss')  # Red line
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.title('Learning Curve of Lattice VAE')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot to a file
        plot_path = os.path.join(model_path, 'learning_curve.png')
        plt.savefig(plot_path)
        print(f'Learning curve saved to {plot_path}')

    # Optionally, display the plot
    # plt.show()
