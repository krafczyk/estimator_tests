import tensorflow as tf
from tensorflow.keras.mixed_precision.experimental import Policy
import tensorflow_datasets as tfds
from modelzoo.common.tf.layers.DenseLayer import DenseLayer
from modelzoo.common.tf.layers.ReshapeLayer import ReshapeLayer
from cerebras.tf.cs_model_to_estimator import KerasModelToModelFn


def input_fn(mode, params={}, input_context=None):
    datasets, info = tfds.load(
        name='mnist',
        with_info=True,
        as_supervised=True)
    mnist_dataset = \
        (datasets['train'] if mode == tf.estimator.ModeKeys.TRAIN else datasets['test'])
    num_examples = \
        (info.splits['train'].num_examples if mode == tf.estimator.ModeKeys.TRAIN else info.splits['test'].num_examples)

    def normalize_img(image, label):
        return tf.cast(image, tf.float16) / 255., tf.cast(label, tf.int32)

    mnist_dataset = mnist_dataset.map(
        normalize_img,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # Data needs to be in channels first format.
    def transpose_img(image, label):
        return tf.transpose(image, perm=[2, 0, 1]), label

    mnist_dataset = mnist_dataset.map(
        transpose_img,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    if input_context:
        mnist_dataset = mnist_dataset.shard(
            input_context.num_input_pipelines,
            input_context.input_pipeline_id)

    batch_size = 32
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Repeat is insisted by CS
        # Drop remainder is insisted by CS as well.
        return mnist_dataset.cache() \
                            .shuffle(num_examples) \
                            .batch(batch_size, drop_remainder=True) \
                            .prefetch(tf.data.experimental.AUTOTUNE) \
                            .repeat()
    else:
        # Repeat is insisted by CS
        # Drop remainder is insisted by CS as well.
        return mnist_dataset.cache() \
                            .batch(batch_size, drop_remainder=True) \
                            .prefetch(tf.data.experimental.AUTOTUNE) \
                            .repeat()


def normalize_img_16(image, label):
    return tf.cast(image, tf.float16) / 255., tf.cast(label, tf.int32)


def normalize_img_32(image, label):
    return tf.cast(image, tf.float32) / 255., tf.cast(label, tf.int32)


def img_channels_first(image, label):
    return tf.transpose(image, perm=[2, 0, 1]), label


def flatten(images, label):
    batch_size = images.shape[0]
    images = tf.reshape(images, [batch_size, -1])
    return images, label


def train_input_fn(params={}, input_context=None):
    # setting num_parallel_calls to 0 implies AUTOTUNE
    num_parallel_calls = params.get("num_parallel_calls", 0)
    if num_parallel_calls == 0:
        num_parallel_calls = tf.data.experimental.AUTOTUNE

    datasets, info = tfds.load(
        name='mnist',
        with_info=True,
        as_supervised=True)
    mnist_dataset = datasets['train']
    num_examples = info.splits['train'].num_examples

    mnist_dataset = mnist_dataset.map(
        normalize_img_16,
        num_parallel_calls=num_parallel_calls
    )

    mnist_dataset = mnist_dataset.map(
        img_channels_first,
        num_parallel_calls=num_parallel_calls
    )

    if input_context:
        mnist_dataset = mnist_dataset.shard(
            input_context.num_input_pipelines,
            input_context.input_pipeline_id)

    batch_size = 32
    mnist_dataset = mnist_dataset.cache() \
                                 .shuffle(num_examples) \
                                 .batch(batch_size, drop_remainder=True) \
                                 .prefetch(tf.data.experimental.AUTOTUNE) \
                                 .repeat()

    return mnist_dataset


def test_input_fn(params={}, input_context=None):
    # setting num_parallel_calls to 0 implies AUTOTUNE
    num_parallel_calls = params.get("num_parallel_calls", 0)
    if num_parallel_calls == 0:
        num_parallel_calls = tf.data.experimental.AUTOTUNE

    datasets, info = tfds.load(
        name='mnist',
        with_info=True,
        as_supervised=True)
    mnist_dataset = datasets['test']
    num_examples = info.splits['test'].num_examples

    mnist_dataset = mnist_dataset.map(
        normalize_img_16,
        num_parallel_calls=num_parallel_calls
    )

    mnist_dataset = mnist_dataset.map(
        img_channels_first,
        num_parallel_calls=num_parallel_calls
    )

    if input_context:
        mnist_dataset = mnist_dataset.shard(
            input_context.num_input_pipelines,
            input_context.input_pipeline_id)

    batch_size = 32
    mnist_dataset = mnist_dataset.cache() \
                        .batch(batch_size, drop_remainder=True) \
                        .prefetch(tf.data.experimental.AUTOTUNE) \
                        .repeat()

    return mnist_dataset


def build_model_fn():
    # Try to set float16 policy
    dtype = Policy('mixed_float16', loss_scale=None)
    tf.keras.mixed_precision.experimental.set_policy(dtype)
    #tf.keras.backend.set_floatx('float16')

    model = tf.keras.models.Sequential([
        #ReshapeLayer((28*28*1,), input_shape=(28,28,1)),
        DenseLayer(128, activation='relu'),
        DenseLayer(10, activation='linear')
    ])

    return KerasModelToModelFn(model)


def model_fn(features, labels, mode, params):
    # Try to set float16 policy
    dtype = Policy('mixed_float16', loss_scale=None)
    tf.keras.mixed_precision.experimental.set_policy(dtype)
    #tf.keras.backend.set_floatx('float16')

    #inp = tf.keras.layers.Input((28,28,1))
    #last_layer = inp
    #last_layer = tf.keras.layers.Flatten()(last_layer)
    #last_layer = DenseLayer(128, activation='relu')(last_layer)
    #last_layer = DenseLayer(10, activation='linear')(last_layer)

    model = tf.keras.models.Sequential([
        #tf.keras.layers.Flatten(input_shape=(28,28,1)),
        #ReshapeLayer((28*28*1,), input_shape=(28,28,1)),
        #tf.keras.layers.Dense(128, activation='relu'),
        #tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Conv2D(32, 3, dilation_rate=(2,2), activation='relu', input_shape=(1,28,28), data_format='channels_first'),
        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dense(10, activation='linear')
        tf.keras.layers.Dense(10, activation='linear')
    ])

    #model = tf.keras.Model(
    #   inputs=inp,
    #   outputs=last_layer)

    #def mdl(feat, training=False):
    #    temp = tf.reshape(feat, (28*28*1,))
    #    tf.nn.

    if mode == tf.estimator.ModeKeys.TRAIN:
        logits = model(features, training=True)
        optimizer = tf.compat.v1.train.AdamOptimizer()
        loss_op = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits))
        train_op = optimizer.minimize(
            loss_op, tf.compat.v1.train.get_or_create_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss_op,
            train_op=train_op,
        )
    elif mode == tf.estimator.ModeKeys.PREDICT:
        preds = tf.nn.softmax(model(features, training=True))
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=preds
        )
    elif mode == tf.estimator.ModeKeys.EVAL:
        preds = tf.nn.softmax(model(features, training=True))
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False)(labels, preds)
        acc = tf.compat.v1.metrics.accuracy(labels, tf.argmax(preds, axis=-1))
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops={'acc': acc}
        )
