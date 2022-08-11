import tensorflow as tf
from tensorflow.keras.mixed_precision.experimental import Policy
import tensorflow_datasets as tfds


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
        return tf.cast(image, tf.float32) / 255., tf.cast(label, tf.int32)

    mnist_dataset = mnist_dataset.map(
        normalize_img,
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
        return mnist_dataset.cache().shuffle(num_examples).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE).repeat()
    else:
        # Repeat is insisted by CS
        # Drop remainder is insisted by CS as well.
        return mnist_dataset.cache().batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE).repeat()


def train_input_fn(params={}, input_context=None):
    datasets, info = tfds.load(
        name='mnist',
        with_info=True,
        as_supervised=True)
    mnist_dataset = datasets['train']
    num_examples = info.splits['train'].num_examples

    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255., tf.cast(label, tf.int32)

    mnist_dataset = mnist_dataset.map(
        normalize_img,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    if input_context:
        mnist_dataset = mnist_dataset.shard(
            input_context.num_input_pipelines,
            input_context.input_pipeline_id)

    batch_size = 32
    return mnist_dataset.cache().shuffle(num_examples).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE).repeat()


def test_input_fn(params={}, input_context=None):
    datasets, info = tfds.load(
        name='mnist',
        with_info=True,
        as_supervised=True)
    mnist_dataset = datasets['test']
    num_examples = info.splits['test'].num_examples

    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255., tf.cast(label, tf.int32)

    mnist_dataset = mnist_dataset.map(
        normalize_img,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    if input_context:
        mnist_dataset = mnist_dataset.shard(
            input_context.num_input_pipelines,
            input_context.input_pipeline_id)

    batch_size = 32
    return mnist_dataset.cache().batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE).repeat()


def model_fn(features, labels, mode, params):
    # Try to set float16 policy
    dtype = Policy('mixed_float16', loss_scale=None)
    tf.keras.mixed_precision.experimental.set_policy(dtype)
    tf.keras.backend.set_floatx('float16')

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28,1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='sigmoid')
    ])

    if mode == tf.estimator.ModeKeys.TRAIN:
        preds = model(features, training=True)
        optimizer = tf.compat.v1.train.AdamOptimizer()
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False)(labels, preds)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op = optimizer.minimize(
                loss, tf.compat.v1.train.get_or_create_global_step())
        )
    elif mode == tf.estimator.ModeKeys.PREDICT:
        preds = model(features, training=True)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=preds
        )
    elif mode == tf.estimator.ModeKeys.EVAL:
        preds = model(features, training=True)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False)(labels, preds)
        acc = tf.compat.v1.metrics.accuracy(labels, tf.argmax(preds, axis=-1))
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops={'acc': acc}
        )
