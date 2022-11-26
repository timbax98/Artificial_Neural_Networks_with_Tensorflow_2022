import tensorflow as tf
import tensorflow_datasets as tfds
import datetime

#%load_ext tensorboard




# 1. get mnist from tensorflow_datasets
mnist = tfds.load("mnist", split =["train","test"], as_supervised=True)
train_ds_1 = mnist[0]
val_ds_1 = mnist[1]
train_ds_2 = mnist[0]
val_ds_2 = mnist[1]

# 2. write function to create the dataset that we want
def preprocess1(data, batch_size, task):
    # image should be float
    data = data.map(lambda x, t: (tf.cast(x, float), t))
    # image should be flattened
    data = data.map(lambda x, t: (tf.reshape(x, (-1,)), t))
    # image vector will here have values between -1 and 1
    data = data.map(lambda x,t: ((x/128.)-1., t))
    # we want to have two mnist images in each example
    # this leads to a single example being ((x1,y1),(x2,y2))
    zipped_ds = tf.data.Dataset.zip((data.shuffle(2000),
                                     data.shuffle(2000)))
    if task == 1:
        # map ((x1,y1),(x2,y2)) to (x1,x2, y1==y2*) *boolean
        zipped_ds = zipped_ds.map(lambda x1, x2: (x1[0], x2[0], (x1[1] + x2[1]) >= 5))

    elif task == 2:
        zipped_ds = zipped_ds.map(lambda x1, x2: (x1[0], x2[0], x1[1] - x2[1]))

    # transform boolean target to int
    zipped_ds = zipped_ds.map(lambda x1, x2, t: (x1,x2, tf.cast(t, tf.int32)))
    # batch the dataset
    zipped_ds = zipped_ds.batch(batch_size)
    # prefetch
    zipped_ds = zipped_ds.prefetch(tf.data.AUTOTUNE)
    return zipped_ds

# training an testing datasets for task 1
train_ds_1 = preprocess1(train_ds_1, batch_size=32, task=1) #train_ds.apply(preprocess)
val_ds_1 = preprocess1(val_ds_1, batch_size=32, task = 1)

# training an testing datasets for task 1
train_ds_2 = preprocess1(train_ds_2, batch_size=32, task= 2)  # train_ds.apply(preprocess)
val_ds_2 = preprocess1(val_ds_2, batch_size=32, task= 2)




# check the contents of the dataset
for img1, img2, label in train_ds_1.take(1):
    print(img1.shape, img2.shape, label.shape)


class TwinMNISTModel(tf.keras.Model):

    # 1. constructor
    def __init__(self, task):
        super().__init__()
        # inherit functionality from parent class

        # optimizer, loss function and metrics
        self.metrics_list = [tf.keras.metrics.Mean(name="loss"),
                             tf.keras.metrics.BinaryAccuracy()]

        self.optimizer = tf.keras.optimizers.Adam()
        # Adam optimizer performs a bit better than SGD

        if task ==1:
            self.loss_function = tf.keras.losses.BinaryCrossentropy()
        elif task ==2:
            self.loss_function = tf.keras.losses.MeanSquaredError()

        # layers to be used
        self.dense1 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(32, activation=tf.nn.relu)

        self.out_layer = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    # 2. call method (forward computation)
    def call(self, images, training = False):
        img1, img2 = images

        img1_x = self.dense1(img1)
        img1_x = self.dense2(img1_x)

        img2_x = self.dense1(img2)
        img2_x = self.dense2(img2_x)

        combined_x = tf.concat([img1_x, img2_x], axis=1)

        return self.out_layer(combined_x)

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
        img1, img2, label = data

        with tf.GradientTape() as tape:
            output = self((img1, img2), training=True)
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

        img1, img2, targets = data
        predictions = self((img1, img2), training=False)
        loss = self.loss_function(targets, predictions) + tf.reduce_sum(self.losses)

        self.metrics[0].update_state(loss)
        # for accuracy metrics:
        for metric in self.metrics[1:]:
            metric.update_state(targets, predictions)

        return {m.name: m.result() for m in self.metrics}



# instantiate the model
model = TwinMNISTModel(task= 1)
#model2 = TwinMNISTModel(task = 2)

# run model on input once so the layers are built
#model(tf.keras.Input((32,784),(32,784)));
#model.summary()




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


#run the training loop
training_loop(model=model,
                train_ds=train_ds_1,
                val_ds=val_ds_1,
                epochs=10,
                train_summary_writer=train_summary_writer,
                val_summary_writer=val_summary_writer)

# save the model with a meaningful name
model.save_weights(f"saved_model_{config_name}", save_format="tf")





