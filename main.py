import tensorflow as tf
from tensorflow.keras.layers import Reshape, Dropout, Dense
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import time

# Written by Justin Gallagher
# Last Updated: 04/13/2021
# ImageGAN is a very simple, crudely made, generative adversarial network that I have been playing around
# with to see if I could generate new artificially made images from an image dataset. This is not a finished
# program, and it probably will never be. The goal of this program is to learn how to use Keras and TF. I'm
# surprised it even works, but the output images I get are not anywhere near as good as StyleGAN, so don't
# expect anything great from it.

# Suppress annoying tf output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Generation Resolution - IMAGES HAVE TO BE SQUARE OR ELSE...
# Training data is also scaled to this value.
GENERATE_RES = 3  # Generation Resolution Factor
# Set this factor to match the px value of your images.
# (1=32px, 2=64px, 3=96px, 4=128px, 5=256px, 6=512px, etc.)
# So for example, if using 128x128px images, use 'GENERATE_RES = 4'
GENERATE_SQUARE = 32 * GENERATE_RES  # rows/cols (should be square)
IMAGE_CHANNELS = 3  # Color images

# Values for pre-viewing images (28 sample images per epoch)
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 16

# Size vector to generate images from
SEED_SIZE = 100

# Configuration
DATA_PATH = './'  # Set the directory
EPOCHS = 1000  # The total number of iterations of training this program will do.
BATCH_SIZE = 32  # 32 works best for me
BUFFER_SIZE = 60000  # buffer size


def main():
    print(f"This program will generate {GENERATE_SQUARE}px square images as output.")

    # Preprocessed numpy file used to store training data, can be used again
    training_binary_path = os.path.join(DATA_PATH, f'training_data_{GENERATE_SQUARE}_{GENERATE_SQUARE}.npy')

    print(f"Looking for file: {training_binary_path}")

    # If no training data is found, proceed with loading the images from img_dataset into 'training_data' array
    if not os.path.isfile(training_binary_path):
        start = time.time()
        print("Loading training images...")

        # BulkImageResizer was used for the data set. Inside the main directory of ImageGAN,
        # place the resized images inside the img_dataset folder.
        training_data = []
        resized_cars_path = os.path.join(DATA_PATH, 'img_dataset')
        for filename in tqdm(os.listdir(resized_cars_path)):
            path = os.path.join(resized_cars_path, filename)
            image = Image.open(path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
            training_data.append(np.asarray(image))
        training_data = np.reshape(training_data, (-1, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))
        training_data = training_data.astype(np.float32)
        training_data = training_data / 127.5 - 1.

        # After loading, save data to Numpy preprocessed file and display timer
        print("Saving training image binary...")
        np.save(training_binary_path, training_data)
        elapsed = time.time() - start
        print(f'Image preprocess time: {time_formatter(elapsed)}')
    # Else, training data has been found, load into 'training_data' array
    else:
        print("Loading previous training data...")
        training_data = np.load(training_binary_path)

    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(training_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    # Build the generator model
    generator = build_generator(SEED_SIZE, IMAGE_CHANNELS)

    noise = tf.random.normal([1, SEED_SIZE])
    generated_image = generator(noise, training=False)

    # Define the shape of each image (i.e. (96, 96, 3))
    image_shape = (GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS)

    # Build the discriminator model
    discriminator = build_discriminator(image_shape)
    decision = discriminator(generated_image)

    # This method returns a helper function to compute cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy()

    # Optimizers
    generator_optimizer = tf.keras.optimizers.Adam(1.5e-4, 0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(1.5e-4, 0.5)

    # Begin training process
    train(generator, discriminator, generator_optimizer, discriminator_optimizer, cross_entropy, train_dataset, EPOCHS)

    # Save data
    generator.save(os.path.join(DATA_PATH, "cars_generator.h5"))


# Formatted string used to display Hours, Minutes, and Seconds
def time_formatter(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


def build_generator(seed_size, channels):
    model = Sequential()

    model.add(Dense(4 * 4 * 256, activation="relu", input_dim=seed_size))
    model.add(Reshape((4, 4, 256)))

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    if GENERATE_RES > 1:
        model.add(UpSampling2D(size=(GENERATE_RES, GENERATE_RES)))
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

    model.add(Conv2D(channels, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    return model


def build_discriminator(image_shape):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape,
                     padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
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
    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model


def save_images(generator, cnt, noise):
    image_array = np.full((
        PREVIEW_MARGIN + (PREVIEW_ROWS * (GENERATE_SQUARE + PREVIEW_MARGIN)),
        PREVIEW_MARGIN + (PREVIEW_COLS * (GENERATE_SQUARE + PREVIEW_MARGIN)), 3),
        255, dtype=np.uint8)

    generated_images = generator.predict(noise)

    generated_images = 0.5 * generated_images + 0.5

    image_count = 0
    for row in range(PREVIEW_ROWS):
        for col in range(PREVIEW_COLS):
            r = row * (GENERATE_SQUARE + 16) + PREVIEW_MARGIN
            c = col * (GENERATE_SQUARE + 16) + PREVIEW_MARGIN
            image_array[r:r + GENERATE_SQUARE, c:c + GENERATE_SQUARE] \
                = generated_images[image_count] * 255
            image_count += 1

    output_path = os.path.join(DATA_PATH, 'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filename = os.path.join(output_path, f"train-{cnt}.png")
    im = Image.fromarray(image_array)
    im.save(filename)


def discriminator_loss(cross_entropy, real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(cross_entropy, fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# This annotation causes the function to be "compiled".
@tf.function
def train_step(cross_entropy, images, generator, discriminator, generator_optimizer, discriminator_optimizer):
    seed = tf.random.normal([BATCH_SIZE, SEED_SIZE])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(seed, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(cross_entropy, fake_output)
        disc_loss = discriminator_loss(cross_entropy, real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(
            gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(
            gradients_of_discriminator,
            discriminator.trainable_variables))
    return gen_loss, disc_loss


def train(generator, discriminator, generator_optimizer, discriminator_optimizer, cross_entropy, dataset, epochs):
    # determine a random seed and begin timer for total training time
    fixed_seed = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, SEED_SIZE))
    start = time.time()

    # loop over number of epochs
    for epoch in range(epochs):
        # begin timer for individual epoch training time
        epoch_start = time.time()

        # lists for generator and discriminator losses
        gen_loss_list = []
        disc_loss_list = []

        for image_batch in dataset:
            t = train_step(cross_entropy, image_batch, generator, discriminator,
                           generator_optimizer, discriminator_optimizer)
            gen_loss_list.append(t[0])
            disc_loss_list.append(t[1])

        g_loss = sum(gen_loss_list) / len(gen_loss_list)
        d_loss = sum(disc_loss_list) / len(disc_loss_list)

        epoch_elapsed = time.time() - epoch_start
        print(
            f'Epoch {epoch + 1}, Generator Loss Value = {g_loss}, '
            f'Discriminator Loss Value = {d_loss}, Time = {time_formatter(epoch_elapsed)}')
        save_images(generator, epoch, fixed_seed)

    elapsed = time.time() - start
    print(f'Training time: {time_formatter(elapsed)}')


if __name__ == '__main__':
    main()
