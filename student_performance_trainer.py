
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam.impl as tft_beam
from tfx.components.trainer.fn_args_utils import FnArgs
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx_bsl.public import tfxio
from tfx.examples.penguin import penguin_utils_base as base

# Feature keys
FEATURE_KEYS = [
    'normalized_attendance',
    'normalized_sleep_hours',
    'normalized_socioeconomic_score',
    'normalized_study_hours'
]

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

def build_model():
    """Membangun model Keras."""
    inputs = {key: tf.keras.layers.Input(shape=(1,), name=key) for key in FEATURE_KEYS}
    x = tf.keras.layers.Concatenate()(list(inputs.values()))
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model

def run_fn(fn_args: FnArgs):
    """Fungsi utama Trainer untuk melatih model."""
    
    # Membangun model
    model = build_model()

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Membaca dataset
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, 10)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, 10)

    print(train_dataset.element_spec)
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
