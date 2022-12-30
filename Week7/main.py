import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.ops.array_ops import zeros
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.keras.layers import Dense, Dropout
import datetime
import pprint
import tqdm
import numpy

## Data pipeline

ds_train, ds_test = tfds.load('mnist', split=['train', 'test'], as_supervised=True)
def preprocess(ds, seq_len, batch_size):
  # shuffle
  ds = ds.shuffle(2000)
  # change datatype of image values
  ds = ds.map(lambda x, t: (tf.cast(x, dtype=tf.dtypes.float32), t))
  # normaluze image values
  ds = ds.map(lambda x, t: ((x/128.)-1., t))
  #only get the targets, to keep this demonstration simple (and force students to understand the code if they are using it by rewriting it respectively)
  ds = ds.map(lambda x, t: (x, tf.cast(t, dtype=tf.dtypes.int32)))
  # use window to create subsequences. This means ds is not a dataset of datasets, i.e. every single entry in the dataset is itself a small tf.data.Dataset object with seq_len many entries!
  ds = ds.window(seq_len)
  #make sure to check tf.data.Dataset.scan() to understand how this works!
  def alternating_scan_function(state, elem):
    #state is allways the sign to use!
    old_sign = state
    #just flip the sign for every element
    new_sign = old_sign*-1
    #elem is just the target of the element. We need to apply the appropriate sign to it!
    signed_target = elem*old_sign
    #we need to return a tuple for the scan function: The new state and the output element
    out_elem = signed_target
    new_state = new_sign
    return new_state, out_elem
  #we now want to apply this function via scanning, resulting in a dataset where the signs are alternating
  #remember we have a dataset, where each element is a sub dataset due to the windowing!
  ds = ds.map(lambda x_seq, t_seq: (x_seq, t_seq.scan(initial_state=1, scan_func=alternating_scan_function)))
  #now we need a scanning function which implements a cumulative sum, very similar to the cumsum used above
  def scan_cum_sum_function(state, elem):
    #state is the sum up the the current element, element is the new digit to add to it
    sum_including_this_elem = state+elem
    #both the element at this position and the returned state should just be sum up to this element, saved in sum_including_this_elem
    return sum_including_this_elem, sum_including_this_elem
  #again we want to apply this to the subdatasets via scan, with a starting state of 0 (sum before summing is zero...)
  ds = ds.map(lambda x_seq, t_seq: (x_seq, t_seq.scan(initial_state=0, scan_func=scan_cum_sum_function)))
  #finally we need to create a single element from everything in the subdataset
  ds = ds.map(lambda x_seq, t_seq: (x_seq.batch(seq_len).get_single_element(), t_seq.batch(seq_len).get_single_element()))
  # cache
  ds = ds.cache()
  # batch
  ds = ds.batch(batch_size)
  # prefetch
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds

# define sequence length
SEQ_LEN = 10
# define batch size
BATCH_SIZE = 32

train_dataset = preprocess(ds_train, SEQ_LEN, BATCH_SIZE)
val_dataset = preprocess(ds_test, SEQ_LEN, BATCH_SIZE)

for x, t in train_dataset.take(1):
    print(x.shape, t.shape)



## Model

import dataset


class ourlstm(tf.keras.layers.AbstractRNNCell):

    def __init__(self, units,
                 **kwargs):  # units = units of the weight matrixs in each dense layer = units of the output
        super().__init__(**kwargs)

        self.units = units

        self.forgetgate = tf.keras.layers.Dense(units,
                                                kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0,
                                                                                                    seed=None),
                                                activation='sigmoid')
        self.inputgate1 = tf.keras.layers.Dense(units,
                                                kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0,
                                                                                                    seed=None),
                                                activation='sigmoid')
        self.inputgate2 = tf.keras.layers.Dense(units,
                                                kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0,
                                                                                                    seed=None),
                                                activation='tanh')
        self.outputgate = tf.keras.layers.Dense(units,
                                                kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0,
                                                                                                    seed=None),
                                                activation='sigmoid')

    @property
    def state_size(self):
        return [tf.TensorShape([self.units]),
                tf.TensorShape([self.units])]

    @property
    def output_size(self):
        return [tf.TensorShape([self.units])]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return [tf.zeros([batch_size, self.units]),
                tf.zeros([batch_size, self.units])]

    # The LSTM-cell layer’s call method should take one (batch of) feature vector(s) as its input,
    # along with the ”states”, a list containing the different state tensors of the LSTM cell (cell state and hidden state!).
    def call(self, inputs, states):
        # unpack the states
        cell_s = states[0]
        hidden_s = states[1]

        concat_value = tf.concat([inputs, hidden_s], axis=-1)  # Leon: or use a tuple

        x1 = self.forgetgate(concat_value)
        x1 = tf.math.multiply(x1, cell_s)  # or use *

        x2 = self.inputgate1(concat_value)
        x3 = self.inputgate2(concat_value)

        x3 = tf.math.multiply(x2, x3)
        new_cell_s = tf.math.add(x1, x3)

        x4 = self.outputgate(concat_value)
        new_hidden_s = tf.math.multiply(x4, tf.math.tanh(new_cell_s))

        return new_hidden_s, (new_hidden_s, new_cell_s)
        # The returns should be the output of the LSTM, to be used to compute the model
        # output for this time-step (usually the hidden state), as well as a list containing
        # the new states (e.g. [new hidden state, new cell state])

    def get_config(self):
        return {"hidden_units": self.units}


class BasicCNN_LSTM(tf.keras.Model):
    def __init__(self):
        super(BasicCNN_LSTM, self).__init__()

        self.convlayer1 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu',
                                                 batch_input_shape=(dataset.batch_size, dataset.sequence_len, 28, 28,
                                                                    1))  # , input_shape=(28, 28, 1))
        self.convlayer2 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu',
                                                 batch_input_shape=(dataset.batch_size, dataset.sequence_len, 28, 28,
                                                                    1))  # , input_shape=(28, 28, 1))
        self.convlayer3 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu',
                                                 batch_input_shape=(dataset.batch_size, dataset.sequence_len, 28, 28,
                                                                    1))  # , input_shape=(28, 28, 1))
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        self.global_pool = tf.keras.layers.GlobalAvgPool2D()
        self.timedist = tf.keras.layers.TimeDistributed(self.global_pool)

        self.rnn = tf.keras.layers.RNN(ourlstm(8), return_sequences=True)
        self.batchnorm2 = tf.keras.layers.BatchNormalization()

        self.outputlayer = tf.keras.layers.Dense(units=1, activation=None)

    @tf.function  # Leon: comment it out when debugging
    def call(self, x):
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.convlayer3(x)

        x = self.batchnorm1(x)
        x = self.timedist(x)  # Here, the shape should be (bs, sequence-length, features) before LSTM

        x = self.rnn(x)

        x = self.batchnorm2(x)
        x = self.outputlayer(x)

        return x

    @property
    def metrics(self):
        return self.metrics_list

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()

    def train_step(self, data):
        x, t = data

        with tf.GradientTape() as tape:
            output = self(x, training=True)
            loss = self.loss_function(t, output)

        gradients = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # update loss metric
        self.metrics[0].update_state(loss)
        for metric in self.metrics[1:]:
            metric.update_state(t, output)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        img, label = data

        output = self(img, training=False)
        loss = self.loss_function(label, output)

        # update loss metric
        self.metrics[0].update_state(loss)
        for metric in self.metrics[1:]:
            metric.update_state(label, output)
        return {m.name: m.result() for m in self.metrics}


model = BasicCNN_LSTM

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

"""def test_once(model, train_ds, val_ds, val_summary_writer):
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
    print("\n")"""



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
#test_once(model= BasicCNN_LSTM, train_ds = train_dataset, val_ds= val_dataset, val_summary_writer=val_summary_writer)
#run the training loop
training_loop(model=BasicCNN_LSTM,
                train_ds=train_dataset,
                val_ds=val_dataset,
                epochs=10,
                train_summary_writer=train_summary_writer,
                val_summary_writer=val_summary_writer)

# save the model with a meaningful name
BasicCNN_LSTM.save_weights(f"saved_model_{config_name}", save_format="tf")