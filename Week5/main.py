import tensorflow_datasets as tfds
import tensorflow as tf
import datetime
import numpy as np
train_ds, test_ds = tfds.load('cifar10', split=['train', 'test'], as_supervised=True)

def prepare_cifar10_data(cifar10):
  #convert data from uint8 to float32
  cifar10 = cifar10.map(lambda img, target: (tf.cast(img, tf.float32), target))
  #sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
  cifar10 = cifar10.map(lambda img, target: ((img/128.)-1., target))
  #create one-hot targets
  cifar10 = cifar10.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
  #cache this progress in memory, as there is no need to redo it; it is deterministic after all
  cifar10 = cifar10.cache()
  #shuffle, batch, prefetch
  cifar10 = cifar10.shuffle(1000)
  cifar10 = cifar10.batch(32)
  cifar10 = cifar10.prefetch(20)
  #return preprocessed dataset
  return cifar10

train_dataset = train_ds.apply(prepare_cifar10_data)
test_dataset = test_ds.apply(prepare_cifar10_data)

def try_model(model, ds):
  for x, t in ds.take(5):
    y = model(x)


from tensorflow.keras.layers import Dense

class BasicConv(tf.keras.Model):
    def __init__(self):
        super(BasicConv, self).__init__()
        self.metrics_list = [tf.keras.metrics.Mean(name="loss"),
                             tf.keras.metrics.BinaryAccuracy()]

        self.optimizer = tf.keras.optimizers.Adam()
        # Adam optimizer performs a bit better than SGD
        self.loss_function = tf.keras.losses.CategoricalCrossentropy()

        #Layers: 2 convlayers then pooling, then 2 more convlayers then pooling

        self.convlayer1 = tf.keras.layers.Conv2D(filters=24, kernel_size=3, padding='same', activation='relu')
        self.convlayer2 = tf.keras.layers.Conv2D(filters=24, kernel_size=3, padding='same', activation='relu')
        self.pooling = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)

        self.convlayer3 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu')
        self.convlayer4 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu')
        self.global_pool = tf.keras.layers.GlobalAvgPool2D()

        self.out = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.pooling(x)
        x = self.convlayer3(x)
        x = self.convlayer4(x)
        x = self.global_pool(x)
        x = self.out(x)
        return x

    # 3. metrics property
    @property
    def metrics(self):
        return self.metrics_list
        # return a list with all metrics in the model

    # 4. reset all metrics objects
    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()

    # 5. train step method
    @tf.function
    def train_step(self, data):
        image, label = data

        with tf.GradientTape() as tape:
            output = self(image, training=True)
            loss = self.loss_function(label, output)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


        # update loss metric
        self.metrics_list[0].update_state(loss)

        # for all metrics except loss, update states (accuracy etc.)
        for metric in self.metrics_list[1:]:
            metric.update_state(label, output)

        # Return a dictionary mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):

        image, targets = data
        predictions = self(image, training=False)
        loss = self.loss_function(targets, predictions) + tf.reduce_sum(self.losses)

        self.metrics[0].update_state(loss)
        # for accuracy metrics:
        for metric in self.metrics[1:]:
            metric.update_state(targets, predictions)

        return {m.name: m.result() for m in self.metrics}

basic_model = BasicConv()
try_model(basic_model, train_dataset)

# Define where to save the log
config_name= "config_name"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

train_log_path = f"logs/{config_name}/{current_time}/train"
val_log_path = f"logs/{config_name}/{current_time}/val"

# log writer for training metrics
train_summary_writer = tf.summary.create_file_writer(train_log_path)

# log writer for validation metrics
val_summary_writer = tf.summary.create_file_writer(val_log_path)

import pprint
import tqdm

def test_once(model, train_ds, val_ds, val_summary_writer):
    for data in train_ds:
        metrics = model.test_step(data)

        # logging the validation metrics to the log file which is used by tensorboard
        with val_summary_writer.as_default():
            for metric in model.metrics:
                tf.summary.scalar(f"{metric.name}", metric.result(), step = 1)

    print([f"val_{key}: {value.numpy()}" for (key, value) in metrics.items()])

    # reset all metrics
    model.reset_metrics()
    print("\n")

    for data in val_ds:
        metrics = model.test_step(data)

        # logging the validation metrics to the log file which is used by tensorboard
        with val_summary_writer.as_default():
            for metric in model.metrics:
                tf.summary.scalar(f"{metric.name}", metric.result(), step = 1)

    print([f"val_{key}: {value.numpy()}" for (key, value) in metrics.items()])

    # reset all metrics
    model.reset_metrics()
    print("\n")



def training_loop(model, train_ds, val_ds, epochs, train_summary_writer, val_summary_writer):
    for epoch in range(epochs):
        print(f"Epoch {epoch}:")

        # Training:

        for data in tqdm.tqdm(train_ds, position=0, leave=True):
            metrics = model.train_step(data)

            # logging the validation metrics to the log file which is used by tensorboard
            with train_summary_writer.as_default():
                for metric in model.metrics:
                    tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)

        # print the metrics
        # print(metrics.shape)
        print([f"{key}: {value.numpy()}" for (key, value) in metrics.items()])

        # reset all metrics (requires a reset_metrics method in the model)
        model.reset_metrics()

        # Validation:
        for data in val_ds:
            metrics = model.test_step(data)

            # logging the validation metrics to the log file which is used by tensorboard
            with val_summary_writer.as_default():
                for metric in model.metrics:
                    tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)

        print([f"val_{key}: {value.numpy()}" for (key, value) in metrics.items()])

        # reset all metrics
        model.reset_metrics()
        print("\n")

import matplotlib.pyplot as plt



# test once
test_once(model=basic_model, train_ds = train_dataset, val_ds= test_dataset, val_summary_writer=val_summary_writer)
#run the training loop
training_loop(model=basic_model,
                train_ds=train_dataset,
                val_ds=test_dataset,
                epochs=10,
                train_summary_writer=train_summary_writer,
                val_summary_writer=val_summary_writer)

# save the model with a meaningful name
basic_model.save_weights(f"saved_model_{config_name}", save_format="tf")
