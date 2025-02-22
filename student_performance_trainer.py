
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam.impl as tft_beam
from tfx.components.trainer.fn_args_utils import FnArgs
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx_bsl.public import tfxio
from tfx.examples.penguin import penguin_utils_base as base

LABEL_KEY = 'normalized_grades'

def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=32) -> tf.data.Dataset:
    # Get post_transform feature spec
    transform_feature_spec = (tf_transform_output.transformed_feature_spec().copy())
    
    # create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=LABEL_KEY)

    return dataset

def build_model(hp, tf_transform_output):
    """Membangun model dengan hyperparameters dari Tuner."""

    feature_spec = tf_transform_output.transformed_feature_spec().copy()

    inputs = {feature: tf.keras.layers.Input(shape=(1,), name=feature) for feature in feature_spec.keys() if feature != LABEL_KEY}
    concatenated = tf.keras.layers.Concatenate()(list(inputs.values()))  # Gabungkan semua fitur

    x = concatenated
    for i in range(hp["num_layers"]):
        x = tf.keras.layers.Dense(
            units=hp[f"units_{i}"],
            activation="relu"
        )(x)

    outputs = tf.keras.layers.Dense(1, activation="linear")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Hyperparameter: Learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp["learning_rate"]
        ),
        loss="mse",
        metrics=["mae"]
    )
    
    model.summary()

    return model
    
def run_fn(fn_args: FnArgs):
    """Fungsi utama Trainer untuk melatih model."""
    print(f"Path file artifact{fn_args.transform_graph_path}")

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    # Membangun model
    model = build_model(fn_args.hyperparameters["values"], tf_transform_output)

    # Membaca dataset
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, 10)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, 10)

    # Callback untuk menyimpan model terbaik
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        fn_args.serving_model_dir,
        save_best_only=True
    )

    # Melatih model
    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        epochs=10,
        callbacks=[model_checkpoint]
    )

    # Menyimpan model akhir
    model.save(fn_args.serving_model_dir, save_format='tf')
