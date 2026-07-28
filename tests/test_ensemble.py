import os
import sys
import tensorflow as tf
import numpy as np
import pytest

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)

from tensorcircuit.applications.ai.ensemble import bagging


@pytest.mark.xfail(
    tuple(int(v) for v in tf.__version__.split(".")[:2]) >= (2, 16),
    reason="legacy optimizer fails tf>=2.16",
)
def test_ensemble_bagging():
    data_amount = 100  # Amount of data to be used
    linear_dimension = 4  # linear demension of the data
    epochs = 10
    batch_size = 32
    lr = 1e-3

    x_train, y_train = (
        np.ones([data_amount, linear_dimension]),
        np.ones([data_amount, 1]),
    )

    obj_bagging = bagging()

    def model():
        DROP = 0.1

        activation = "selu"
        inputs = tf.keras.Input(shape=(linear_dimension,), name="digits")
        x0 = tf.keras.layers.Dense(
            1,
            kernel_regularizer=tf.keras.regularizers.l2(9.613e-06),
            activation=activation,
        )(inputs)
        x0 = tf.keras.layers.Dropout(DROP)(x0)

        x = tf.keras.layers.Dense(
            1,
            kernel_regularizer=tf.keras.regularizers.l2(1e-07),
            activation="sigmoid",
        )(x0)

        model = tf.keras.Model(inputs, x)

        return model

    obj_bagging.append(model(), False)
    obj_bagging.append(model(), False)
    obj_bagging.append(model(), False)
    obj_bagging.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.legacy.Adam(lr),
        metrics=[tf.keras.metrics.AUC(), "acc"],
    )
    obj_bagging.train(
        x=x_train, y=y_train, epochs=epochs, batch_size=batch_size, verbose=0
    )

    v_weight = obj_bagging.predict(x_train, "weight")
    v_average = obj_bagging.predict(x_train, "average")
    v_most = obj_bagging.predict(x_train, "most")
    for v in (v_weight, v_average, v_most):
        assert np.asarray(v).shape == (data_amount,)
    validation_data = []
    validation_data.append(obj_bagging.eval([y_train, v_weight], "acc"))
    validation_data.append(obj_bagging.eval([y_train, v_average], "auc"))
    validation_data.append(obj_bagging.eval([y_train, v_most], "acc"))
    for val in validation_data:
        assert 0.0 <= val <= 1.0
    # x_train and y_train are trivially separable (all ones -> all ones),
    # so a fitted ensemble should beat random guessing on accuracy.
    assert validation_data[0] > 0.5
    assert validation_data[2] > 0.5
