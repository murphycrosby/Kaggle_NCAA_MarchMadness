import logging
import math
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping


early_stopper = EarlyStopping(patience=3)


def compile_model(network, nb_classes, input_shape):
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()

    for i in range(nb_layers):

        if i == 0 and i != (nb_layers - 1):
            model.add(LSTM(nb_neurons[i], input_shape=input_shape, activation=activation[i], return_sequences=True))
        elif i == 0 and i == (nb_layers - 1):
            model.add(LSTM(nb_neurons[i], input_shape=input_shape, activation=activation[i], return_sequences=False))
        elif i < (nb_layers - 1):
            model.add(LSTM(nb_neurons[i], activation=activation[i], return_sequences=True))
        else:
            model.add(LSTM(nb_neurons[i], activation=activation[i], return_sequences=False))

    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def train_and_score(network, nb_classes, x, y):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=1100)

    model = compile_model(network, nb_classes, x_train.shape[1:])

    logging.info('')
    logging.info(network)
    
    model.fit(x_train, y_train,
              batch_size=10,
              epochs=10000,  # using early stopping, so no real limit
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[early_stopper])

    score = model.evaluate(x_test, y_test, verbose=0)
    acc = score[1]
    if math.isnan(acc):
        acc = 100
    logging.info("Score %.2f%%" % acc)

    return acc  # 1 is accuracy. 0 is loss.
