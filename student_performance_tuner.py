
import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.tuner.component import TunerFnResult

NUM_EPOCHS = 10
LABEL_KEY="normalized_grades"

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
        label_key=LABEL_KEY
    )

    return dataset

def build_model(hp, feature_spec):
    inputs = {feature: tf.keras.layers.Input(shape=(1,), name=feature) for feature in feature_spec.keys() if feature != LABEL_KEY}
    concatenated = tf.keras.layers.Concatenate()(list(inputs.values()))

    x = concatenated
    for i in range(hp.Int("num_layers", 1, 3)):
        x = tf.keras.layers.Dense(
            units=hp.Int(f"units_{i}", min_value=32, max_value=128, step=32),
            activation="relu"
        )(x)

    outputs = tf.keras.layers.Dense(1, activation="linear")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
        ),
        loss="mse",
        metrics=["mae"]
    )
    
    return model

# Fungsi tuner untuk pipeline TFX
def tuner_fn(fn_args):
    # Load data dari transform
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    feature_spec = tf_transform_output.transformed_feature_spec()

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, NUM_EPOCHS, batch_size=32)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, NUM_EPOCHS, batch_size=32)

    tuner = kt.Hyperband(
        hypermodel=lambda hp: build_model(hp, feature_spec),
        objective='val_mae',
        max_epochs=NUM_EPOCHS,
        factor=3,
        executions_per_trial=1,
        directory=fn_args.working_dir,
        project_name='student_performance_tuning'
    )
    
    tuner.search(
        train_dataset,
        validation_data=eval_dataset,
        epochs=NUM_EPOCHS,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
    )
        
    return TunerFnResult(
        tuner=tuner, 
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
    })
