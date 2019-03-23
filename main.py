from implementations import ann
from implementations import keras_implementation as ki
from utilities import data_util as du


def keras_execution(x, y, x_test, y_test, learning_rate=1, epochs=30):
    ki.lib_model(x, y, x_test, y_test, learning_rate, epochs)


def custom_execution(x, y, x_test, y_test, learning_rate=1, epochs=10000):
    ann.ann_main(x, y, x_test, y_test, learning_rate, epochs)


def main_init(option):
    max_range = 10
    learning_rate = 1
    keras_epochs, epochs = 100, 10000

    (x, y) = du.load_data(max_range)
    (x, y) = du.shuffle_data(x, y)

    (x_test, y_test) = du.load_data(max_range, single_class_side_number=24)
    (x_test, y_test) = du.shuffle_data(x_test, y_test)

    if 'keras' in option:
        keras_execution(x, y, x_test, y_test, learning_rate=learning_rate, epochs=keras_epochs)

    if 'custom' in option:
        custom_execution(x, y, x_test, y_test, learning_rate=learning_rate, epochs=epochs)


if __name__ == "__main__":
    # option:
    # 'keras'
    # 'custom'
    # ('keras', 'custom')
    main_init(('keras', 'custom'))

