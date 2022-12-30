import tensorflow_datasets as tfds
import tensorflow as tf



(train_ds, val_ds), ds_info = tfds.load('mnist', split=['train', 'test'], as_supervised=True, with_info=True)

# print("ds_info: \n", ds_info)
# tfds.show_examples(train_ds, ds_info)

batch_size = 64
sequence_len = 6

def preprocess(dataset, batchsize, sequence_len):


    # convert data from uint8 to float32
    dataset = dataset.map(lambda img, target: (tf.cast(img, tf.float32), target))

    # input normalization
    dataset = dataset.map(lambda img, target: ((img / 128.) - 1., target))

    # The output of that lambda function should be a tuple of two tensors of shapes (num_images, height, width, 1) and (num_images, 1) or (num_images,)

    # Step 2 - Sequence Batching, Create Targets, Shuffling, Batching & Prefetching
    dataset = dataset.batch(sequence_len)

    # change the target
    # alternate positive, negative target values
    range_vals = tf.range(sequence_len)

    dataset = dataset.map(lambda img, target:
                          (img, tf.where(tf.math.floormod(range_vals,2)==0, target, -target)))

    dataset = dataset.map(lambda img, target:
                          (img, (tf.math.cumsum(target))))

    # cache
    dataset = dataset.cache()

    # shuffle, batch, prefetch
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batchsize)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # The shape of the tensors should be (batch, sequence-length, features).
    return dataset


def expand_dimension(x, y):
    return x, tf.expand_dims(y, axis=-1)


train_ds = preprocess(train_ds, batch_size, 6)
val_ds = preprocess(val_ds, batch_size, 6)

train_ds = train_ds.map(expand_dimension)
val_ds = train_ds.map(expand_dimension)





class lstm(tf.keras.layers.AbstractRNNCell):

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


    def call(self, inputs, states):
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
                                                 batch_input_shape=(batch_size, sequence_len, 28, 28,
                                                                    1))  # , input_shape=(28, 28, 1))
        self.convlayer2 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu',
                                                 batch_input_shape=(batch_size, sequence_len, 28, 28,
                                                                    1))  # , input_shape=(28, 28, 1))
        self.convlayer3 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu',
                                                 batch_input_shape=(batch_size, sequence_len, 28, 28,
                                                                    1))  # , input_shape=(28, 28, 1))
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        self.global_pool = tf.keras.layers.GlobalAvgPool2D()
        self.timedist = tf.keras.layers.TimeDistributed(self.global_pool)

        self.rnn = tf.keras.layers.RNN(lstm(8), return_sequences=True)
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


import matplotlib.pyplot as plt
import datetime as datetime
from pathlib import Path

# Initiate epochs and learning rate as global variables
epochs = 15
lr = 1e-2

mymodel = BasicCNN_LSTM()

loss = tf.keras.losses.MeanSquaredError()
opti = tf.keras.optimizers.Adam(learning_rate=lr)

mymodel.compile(loss=loss,
                optimizer=opti,
                metrics=['MAE'])  # for accuracy - instead of tf.keras.metrics.MeanAbsoluteError()

# save logs with Tensorboard
EXPERIMENT_NAME = "CNN_LSTM"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logging_callback = tf.keras.callbacks.TensorBoard(log_dir=f".Homework/07/logs/{EXPERIMENT_NAME}/{current_time}")

history = mymodel.fit(train_ds,
                      validation_data=val_ds,
                      epochs=epochs,
                      batch_size=batch_size,
                      callbacks=[logging_callback])

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.plot(history.history["MAE"])
plt.plot(history.history["val_MAE"])
plt.legend(labels=["train_loss", "val_loss", "train_error(acc)", "val_error(acc)"])
plt.xlabel("Epoch")
plt.ylabel("MSE(loss), MAE(acc)")
plt.savefig(f"testing: e={epochs},lr={lr}.png")
plt.show()

# save configs (e.g. hyperparameters) of your settings
hw_directory = str(Path(__file__).parents[0])
model_folder = 'my_model07'

dir = hw_directory + '/' + model_folder

mymodel.save(dir)

# checkpoint your model’s weights (or even the complete model)
# mymodel.load_weights(checkpoint_filepath)
# try out checkpoint when there's time:
# checkpoint_filepath = 'checkpoint.hdf5'
# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
#                                                                 save_weights_only=True,
#                                                                 monitor='val_accuracy',
# save_best_only=True)


from pathlib import Path

# getting the directory
hw_directory = str(Path(__file__).parents[0])
model_folder = 'my_model07'
dir = hw_directory + '/' + model_folder

# loading the model
new_model = tf.keras.models.load_model(dir, custom_objects={"lstm": lstm,
                                                            "BasicCNN_LSTM": BasicCNN_LSTM})

# model summary
new_model.summary()