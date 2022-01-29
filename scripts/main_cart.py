###################################################
#   Author: Daniel Adamkowski                     #
###################################################

import sys
from random import randrange

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from scripts.CartSurrogateSplitters import CartSurrogateSplitters
from scripts.DataUtils import create_missing_data, FillDataStrategy, fill_missing_data, load_data


def main(dataset_name):
    dataset_name = "wine"
    dataset = load_data(dataset_name)
    dataset_data, dataset_target = dataset.data, dataset.target

    avg_accuracy = 0
    number_of_iterations = 10
    for n in range(number_of_iterations):
        seed = randrange(1, 1000)
        x_train, x_test, y_train, y_test = train_test_split(dataset_data, dataset_target, test_size=.2, random_state=seed)

        create_missing_data(x_train, malformed_rows_percentage=80, malformed_in_row=3)
        fill_missing_data(x_train, FillDataStrategy.MEAN)

        create_missing_data(x_test, malformed_rows_percentage=50, malformed_in_row=1)

        classifier = CartSurrogateSplitters(x_train, y_train)

        predicted_classes = []
        for test_index in range(len(y_test)):
            predicted_classes.append(classifier.predict(x_test[test_index]))
        avg_accuracy += accuracy_score(y_test, predicted_classes)

    print(avg_accuracy / number_of_iterations)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Dataset needs to be specified as script parameter')
        sys.exit(1)

    main(sys.argv[1])
