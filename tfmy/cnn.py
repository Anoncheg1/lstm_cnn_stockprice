import tensorflow as tf
from tensorflow.keras import Model


class CNNtime(Model):
    _L2_WEIGHT_DECAY = 1e-4

    @staticmethod
    def _gen_l2_regularizer(use_l2_regularizer=True):
        return tf.keras.regularizers.l2(CNNtime._L2_WEIGHT_DECAY) if use_l2_regularizer else None

    @staticmethod
    def _identity_block(input, filters, use_l2_regularizer):
        filter1, filter2, filter3 = filters

        x = tf.keras.layers.Conv2D(
            filters=filter1,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=CNNtime._gen_l2_regularizer(use_l2_regularizer),
            data_format='channels_first')(input)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(
            filters=filter2,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=CNNtime._gen_l2_regularizer(use_l2_regularizer),
            data_format='channels_first')(x)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(
            filters=filter3,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=CNNtime._gen_l2_regularizer(use_l2_regularizer),
            data_format='channels_first')(x)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.add([x, input])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    @staticmethod
    def _conv_block(input, filters, stride, use_l2_regularizer):
        filter1, filter2, filter3 = filters

        x = tf.keras.layers.Conv2D(
            filters=filter1,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=CNNtime._gen_l2_regularizer(use_l2_regularizer),
            data_format='channels_first')(input)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(
            filters=filter2,
            kernel_size=3,
            strides=stride,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=CNNtime._gen_l2_regularizer(use_l2_regularizer),
            data_format='channels_first')(x)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(
            filters=filter3,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=CNNtime._gen_l2_regularizer(use_l2_regularizer),
            data_format='channels_first')(x)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.Activation('relu')(x)

        shortcut = tf.keras.layers.Conv2D(
            filters=filter3,
            kernel_size=1,
            strides=stride,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=CNNtime._gen_l2_regularizer(use_l2_regularizer),
            data_format='channels_first')(input)
        shortcut = tf.keras.layers.BatchNormalization(axis=1)(shortcut)

        x = tf.keras.layers.add([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def __init__(self, global_batch_size, img_size, learning_rate=3e-4, use_l2_regularizer=True):
        """

        :param global_batch_size: for print
        :param img_size: NCHW N: number of images in the batch H: height of the image W: width of the image C: number of channels of the image (ex: 3 for RGB, 1 for grayscale...)
        :param learning_rate:
        :param use_l2_regularizer:
        """
        super(CNNtime, self).__init__()
        self.img_size = img_size
        self.learning_rate = learning_rate
        self.global_batch_size = global_batch_size
        self.use_l2_regularizer = use_l2_regularizer

        # image is HWC (normally e.g. RGB image) however data needs to be NCHW for network
        self.inputs = tf.keras.Input(shape=(img_size[2], None, None))
        # self.inputs = tf.keras.Input(shape=(img_size[2], img_size[0], img_size[1]))
        self.model = self._build_model()

        self.loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _build_model(self):
        # reinterpreted from: https://github.com/tensorflow/models/blob/master/official/vision/image_classification/resnet_model.py

        x = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=7,
            strides=2,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=self._gen_l2_regularizer(self.use_l2_regularizer),
            data_format='channels_first')(self.inputs)
        x = tf.keras.layers.BatchNormalization(
            axis=1)(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        x = self._conv_block(x, [64, 64, 256], stride=1, use_l2_regularizer=self.use_l2_regularizer)
        x = self._identity_block(x, [64, 64, 256],
                                     use_l2_regularizer=self.use_l2_regularizer)
        x = self._identity_block(x, [64, 64, 256],
                                     use_l2_regularizer=self.use_l2_regularizer)

        x = self._conv_block(x, [128, 128, 512], stride=2,
                                 use_l2_regularizer=self.use_l2_regularizer)
        x = self._identity_block(x, [128, 128, 512],
                                     use_l2_regularizer=self.use_l2_regularizer)
        x = self._identity_block(x, [128, 128, 512],
                                     use_l2_regularizer=self.use_l2_regularizer)
        x = self._identity_block(x, [128, 128, 512],
                                     use_l2_regularizer=self.use_l2_regularizer)

        x = self._conv_block(x, [256, 256, 1024], stride=2,
                                 use_l2_regularizer=self.use_l2_regularizer)
        x = self._identity_block(x, [256, 256, 1024],
                                     use_l2_regularizer=self.use_l2_regularizer)
        x = self._identity_block(x, [256, 256, 1024],
                                     use_l2_regularizer=self.use_l2_regularizer)
        x = self._identity_block(x, [256, 256, 1024],
                                     use_l2_regularizer=self.use_l2_regularizer)
        x = self._identity_block(x, [256, 256, 1024],
                                     use_l2_regularizer=self.use_l2_regularizer)
        x = self._identity_block(x, [256, 256, 1024],
                                     use_l2_regularizer=self.use_l2_regularizer)

        x = self._conv_block(x, [512, 512, 2048], stride=2,
                                 use_l2_regularizer=self.use_l2_regularizer)
        x = self._identity_block(x, [512, 512, 2048],
                                     use_l2_regularizer=self.use_l2_regularizer)
        x = self._identity_block(x, [512, 512, 2048],
                                     use_l2_regularizer=self.use_l2_regularizer)

        # output_layer_name5 is tensor with shape <batch_size>, 2048, <img_size>/32, <img_size>/32
        # downsample_factor = 32
        rm_axes = [2, 3]
        x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, rm_axes), name='reduce_mean')(x)

        logits = tf.keras.layers.Dense(
            1,
            kernel_initializer='he_normal',
            kernel_regularizer=self._gen_l2_regularizer(self.use_l2_regularizer),
            bias_regularizer=self._gen_l2_regularizer(self.use_l2_regularizer),
            activation=None,
            name='logits')(x)

        CNNtime = tf.keras.Model(self.inputs, logits, name='CNNtime')

        return CNNtime

    def get_keras_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer

    def set_learning_rate(self, learning_rate):
        self.optimizer.learning_rate = learning_rate

    def get_learning_rate(self):
        return self.optimizer.learning_rate

    def train_step(self, inputs):
        (images, labels, loss_metric) = inputs
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables autodifferentiation.
        with tf.GradientTape() as tape:
            logits = self.model(images, training=True)

            loss_value = self.loss_fn(labels, logits)  # [Nx1]
            # average across the batch (N) with the appropriate global batch size
            loss_value = tf.reduce_sum(loss_value, axis=0) / self.global_batch_size

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, self.model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        loss_metric.update_state(loss_value)

        return loss_value

    @tf.function
    def dist_train_step(self, dist_strategy, inputs):
        per_gpu_loss = dist_strategy.experimental_run_v2(self.train_step, args=(inputs,))
        loss_value = dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_gpu_loss, axis=None)

        return loss_value

    def test_step(self, inputs):
        (images, labels, loss_metric) = inputs
        logits = self.model(images, training=False)

        loss_value = self.loss_fn(labels, logits)
        # average across the batch (N) with the approprite global batch size
        loss_value = tf.reduce_sum(loss_value, axis=0) / self.global_batch_size

        loss_metric.update_state(loss_value)

        return loss_value

    @tf.function
    def dist_test_step(self, dist_strategy, inputs):
        per_gpu_loss = dist_strategy.experimental_run_v2(self.test_step, args=(inputs,))
        loss_value = dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_gpu_loss, axis=None)
        return loss_value


if __name__ == '__main__':
    s = CNNtime(50, [0, 0, 1])

    # CUDA
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        # try:
        #     # Currently, memory growth needs to be the same across GPUs
        #     for gpu in gpus:
        #         tf.config.experimental.set_memory_growth(gpu, True)
        #     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        #     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        # except RuntimeError as e:
        #     # Memory growth must be set before GPUs have been initialized
        #     print(e)
        # Restrict TensorFlow to only allocate 5GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

        # disable logger
        # logging.getLogger('tensorflow').disabled = True

    import numpy as np
    s.build(s.inputs)
    s.summary()
