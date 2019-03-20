from implementations import ann
from implementations import keras_implementation as ki
from utilities import data_util as du

max_range = 10

(x, y) = du.load_data(max_range)
(x, y) = du.shuffle_data(x, y)

(x_test, y_test) = du.load_data(max_range, single_class_side_number=24)
(x_test, y_test) = du.shuffle_data(x_test, y_test)

# ki.lib_model(x, y, x_test, y_test, learning_rate=1, epochs=30)
ann.ann_main(x, y, x_test, y_test, epochs = 10000)

