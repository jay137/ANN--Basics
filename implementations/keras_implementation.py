from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

from utilities import data_util as du


def lib_model(x, y, x_test, y_test, learning_rate=0.2, epochs=100):
    model = Sequential()
    model.add(Dense(2, input_dim=2))
    model.add(Activation('sigmoid'))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(x, y, batch_size=1, nb_epoch=epochs)


# Driver code
if __name__ == "__main__":
    learning_rate = 0.2
    max_range = 10

    (x, y) = du.load_data(max_range)
    (x, y) = du.shuffle_data(x, y)

    (x_test, y_test) = du.load_data(max_range, single_class_side_number=24)
    (x_test, y_test) = du.shuffle_data(x_test, y_test)

    lib_model(x, y, x_test, y_test, learning_rate)
