import tensorflow as tf
import tensorflow_datasets as tfds


def input_fn(mode, input_context=None):
    datasets, info = tfds.load(
        name='mnist',
        with_info=True,
        as_supervised=True)
    mnist_dataset = \
        (datasets['train'] if mode == tf.estimator.ModeKeys.TRAIN else datasets['test'])
    num_examples = \
        (info.splits['train'].num_examples if mode == tf.estimator.ModeKeys.TRAIN else info.splits['test'].num_examples)

    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255., label

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
        return mnist_dataset.cache().shuffle(num_examples).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    else:
        return mnist_dataset.cache().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


def model_fn(features, labels, mode):
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='./mnist_estimator_mdl_test', help="Directory to store model", type=str)

    args = parser.parse_args()

    model_dir = args.model_dir

    session_config = tf.compat.v1.ConfigProto()
    # restrict the estimator to running on certain gpus.
    session_config.gpu_options.visible_device_list = "1"

    estimator_config = tf.estimator.RunConfig(
        model_dir=model_dir,
        session_config=session_config)

    mdl_est = tf.estimator.Estimator(
        model_fn=model_fn,
        config=estimator_config
    )

    mdl_est.train(input_fn=input_fn, steps=20000)

    eval_result = mdl_est.evaluate(input_fn=input_fn, steps=1000)

    print(f"Final eval accuracy: {eval_result['acc']}")
