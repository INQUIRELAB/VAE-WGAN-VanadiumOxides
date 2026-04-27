from __future__ import print_function, division
from keras.regularizers import l2
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.constraints import Constraint
import matplotlib.pyplot as plt

import sys

import numpy as np
import os

from keras import backend as K

def min_formation_energy(y_true,y_pred):
    return K.mean(K.exp(y_pred))

def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    # select images
    X = dataset[ix]
    # generate class labels, -1 for 'real'
    y = -np.ones((n_samples, 1))
    return X, y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = generator.predict(x_input)
    # create class labels with 1.0 for 'fake'
    y = np.ones((n_samples, 1))
    return X, y


# clip model weights to a given hypercube
class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return K.clip(weights, -self.clip_value, self.clip_value)

# WGAN Loss function
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

class CCDCGAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 200

        optimizer = Adam(0.0002, 0.5, decay=1e-4)
        opt = RMSprop(lr=0.00005)
        self.discriminator = self.build_discriminator()
        # WGAN: Use Wasserstein loss and RMSprop optimizer
        self.discriminator.compile(loss=wasserstein_loss,
            optimizer=opt,
            metrics=['accuracy'])

        self.generator = self.build_generator()

        self.constrain = self.rebuild_constrain()

        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        self.discriminator.trainable = True

        valid = self.discriminator(img)

        self.constrain.trainable = False

        formation_energy = self.constrain(img)

        self.final_combined = Model(inputs=z,outputs=[valid, formation_energy])

        # WGAN: Use Wasserstein loss for discriminator output
        losses = [wasserstein_loss, min_formation_energy]
        lossWeights = [ 1.0, 0.1]

        self.final_combined.compile(loss=losses, loss_weights=lossWeights, optimizer=optimizer)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        gradients = K.gradients(y_pred, averaged_samples)[0]
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = K.square(1 - gradient_l2_norm)
        return K.mean(gradient_penalty)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        # WGAN: Apply ClipConstraint for weight clipping
        const = ClipConstraint(0.01)

        model = Sequential()

        # WGAN: Apply kernel_constraint to enforce Lipschitz continuity
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same", kernel_regularizer=l2(0.01), kernel_constraint=const))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same", kernel_regularizer=l2(0.01), kernel_constraint=const))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same", kernel_regularizer=l2(0.01), kernel_constraint=const))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        # WGAN: No sigmoid activation - output is unbounded real number
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def build_constrain(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1024,activation='linear'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256,activation='linear'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256,activation='linear'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256,activation='linear'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(64,activation='linear'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1,activation='linear'))
        model.add(LeakyReLU(alpha=0.2))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def rebuild_constrain(self):
        model=self.build_constrain()
        model.load_weights('./calculation/model/formation_energy_reg.h5')
        model.summary()
        model.name="constrain"
        return model

    def train(self, epochs, batch_size=128, save_interval=50,GAN_calculation_folder_path='./calculation/',X_train_name='train_X.npy'):

        Data=np.load(GAN_calculation_folder_path+'train_X.npy')
        train_ratio = 0.9
        split_index = int(len(Data) * train_ratio)
        X_train1, X_test = Data[:split_index], Data[split_index:]

        X_train = []
        for i in range(3):
            # Randomly select samples with replacement
            indices = np.random.choice(len(X_train1), len(X_train1), replace=True)
            X_train.append(X_train1[indices])

        # Concatenate the augmented data
        X_train = np.concatenate(X_train, axis=0)

        print(X_train.shape)

        X_train = np.expand_dims(X_train, axis=3)
        X_test = np.expand_dims(X_test, axis=3)
        from keras.utils import to_categorical
        # WGAN: Use -1/+1 labels instead of 0/1
        valid1 = -np.ones((X_test.shape[0], 1))
        fake1 = np.ones((X_test.shape[0], 1))
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        new_array = np.ones((batch_size,2))
        d1_hist, d2_hist, g_hist = list(), list(), list()
        D_loss_list, Train_accuracy_list, G_loss_list, From_generator_list, From_constrain_list, Test_accuracy_list, Test_loss_list = list(), list(), list(), list(), list(), list(), list()
        for epoch in range(1,1+epochs):
            d_l1, d_l2, gradients = list(), list(), list()
            generated_images = list()
            for _ in range(5):
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                gen_imgs = self.generator.predict(noise)

                if epoch % 500 == 0:
                    generated_images.append(gen_imgs)

                # WGAN: Train discriminator with proper labels
                d_loss_real1 = self.discriminator.train_on_batch(imgs, valid)
                d_l1.append(d_loss_real1)
                d_loss_fake1 = self.discriminator.train_on_batch(gen_imgs, fake)
                d_l2.append(d_loss_fake1)
            
            if epoch % 500 == 0:
                if not os.path.exists(GAN_calculation_folder_path + 'generated_images'):
                    os.makedirs(GAN_calculation_folder_path + 'generated_images')
                np.save(GAN_calculation_folder_path + f'generated_images_epoch_{epoch}.npy', np.array(generated_images))

            d1_m = [sum(sublist) / len(sublist) for sublist in zip(*d_l1)]
            d1_hist.append(np.mean(d1_m))
            d2_m = [sum(sublist) / len(sublist) for sublist in zip(*d_l2)]
            d2_hist.append(np.mean(d2_m))

            d_loss = 0.5 * np.add(d1_m, d2_m)

            noise1 = np.random.normal(0, 1, (X_test.shape[0], self.latent_dim))
            gen_imgs1 = self.generator.predict(noise1)
            test_loss_discriminator_real, test_accuracy_discriminator_real = self.discriminator.evaluate(X_test, valid1, verbose=0)
            test_loss_discriminator_fake, test_accuracy_discriminator_fake = self.discriminator.evaluate(gen_imgs1, fake1, verbose=0)
            test_accuracy_discriminator = 0.5 * np.add(test_accuracy_discriminator_real, test_accuracy_discriminator_fake)
            test_loss_discriminator = 0.5 * np.add(test_loss_discriminator_real, test_loss_discriminator_fake)

            # Evaluate the constraint part accuracy
            noise_valid = np.random.normal(0, 1, (X_test.shape[0], self.latent_dim))
            _, formation_energy_pred = self.final_combined.predict(noise_valid)

            # WGAN: Train generator with -1 labels (real)
            g_loss = self.final_combined.train_on_batch(noise, [valid,valid])

            D_loss_list.append(d_loss[0])
            Train_accuracy_list.append(100*d_loss[1])
            G_loss_list.append(g_loss[0])
            From_generator_list.append(g_loss[1])
            From_constrain_list.append(g_loss[2])
            Test_accuracy_list.append(test_accuracy_discriminator * 100)
            Test_loss_list.append(test_loss_discriminator)

            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, from generator: %f, from constrain: %f] Test Acc: %.2f%%" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0],g_loss[1],g_loss[2], test_accuracy_discriminator * 100))

            if epoch % save_interval == 0:
                if not os.path.exists(GAN_calculation_folder_path+'step_by_step_GAN_model/'):
                    os.makedirs(GAN_calculation_folder_path+'step_by_step_GAN_model/')
                self.generator.save(GAN_calculation_folder_path+'step_by_step_GAN_model/generator.h5'.format(epoch))
                self.discriminator.save(GAN_calculation_folder_path+'step_by_step_GAN_model/discriminator.h5'.format(epoch))
        np.savetxt(GAN_calculation_folder_path + 'd_loss.txt', D_loss_list)
        np.savetxt(GAN_calculation_folder_path + 'Train_accuracy.txt', Train_accuracy_list)
        np.savetxt(GAN_calculation_folder_path + 'g_loss.txt', G_loss_list)
        np.savetxt(GAN_calculation_folder_path + 'From_generator.txt', From_generator_list)
        np.savetxt(GAN_calculation_folder_path + 'From_constrain.txt', From_constrain_list)
        np.savetxt(GAN_calculation_folder_path + 'Test_accuracy.txt', Test_accuracy_list)
        np.savetxt(GAN_calculation_folder_path + 'Test_loss.txt', Test_loss_list)
