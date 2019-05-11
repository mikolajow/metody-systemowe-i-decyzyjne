# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

from __future__ import division
import numpy as np
import collections


def hamming_distance(X, X_train):
    """
    :param X: zbior porownwanych obiektow N1xD
    :param X_train: zbior obiektow do ktorych porownujemy N2xD
    Funkcja wyznacza odleglosci Hamminga obiektow ze zbioru X od
    obiektow X_train. ODleglosci obiektow z jednego i drugiego
    zbioru zwrocone zostana w postaci macierzy
    :return: macierz odleglosci pomiedzy obiektami z X i X_train N1xN2
    """
    x_decompresed = X.todense().astype(int)
    x_decompresed_negated = (~(X.todense())).astype(int)

    x_train_decompresed_negated = (~(X_train.todense())).astype(int)
    x_train_decompresed = X_train.todense().astype(int)

    #print(x_decompresed.shape)
    #print(x_train_decompresed.shape)

    multiplied = x_decompresed @ (x_train_decompresed.transpose())
    negated_multiplayed = x_decompresed_negated @ (x_train_decompresed_negated.transpose())

    similarity_matrinx = multiplied + negated_multiplayed

    dimensions_matrix = x_decompresed.shape[1] * np.ones(similarity_matrinx.shape)

    #print(dimensions_matrix.shape)

    return dimensions_matrix - similarity_matrinx


def sort_train_labels_knn(Dist, y):
    """
    Funkcja sortujaca etykiety klas danych treningowych y
    wzgledem prawdopodobienstw zawartych w macierzy Dist.
    Funkcja zwraca macierz o wymiarach N1xN2. W kazdym
    wierszu maja byc posortowane etykiety klas z y wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist
    :param Dist: macierz odleglosci pomiedzy obiektami z X
    i X_train N1xN2
    :param y: wektor etykiet o dlugosci N2
    :return: macierz etykiet klas posortowana wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist. Uzyc algorytmu mergesort.
    """
    #print(Dist.shape)
    #print(y)
    result = y[Dist.argsort(kind='mergesort')]
    #print(result)
    return result


def p_y_x_knn(y, k):
    """
    Funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla
    kazdej z klas dla obiektow ze zbioru testowego wykorzystujac
    klasfikator KNN wyuczony na danych trenningowych
    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najblizszuch sasiadow dla KNN
    :return: macierz prawdopodobienstw dla obiektow z X
    """
    # print(y.shape)
    #print(y)
    # print(k)
    categories = set(y[0])
    # print(categories)
    # print(number_of_categories)
    result = []
    for row in y:
        result_row = []
        first_k_indexes = row[0:k]
        for i in range(len(categories)):
            result_row.append(np.count_nonzero(first_k_indexes == i+1))
            # print(result_row)
        result.append(result_row)
    result = np.array(result) / k
    #print(result)
    return result





def classification_error(p_y_x, y_true):
    """
    Wyznacz blad klasyfikacji.
    :param p_y_x: macierz przewidywanych prawdopodobienstw
    :param y_true: zbior rzeczywistych etykiet klas 1xN.
    Kazdy wiersz macierzy reprezentuje rozklad p(y|x)
    :return: blad klasyfikacji
    """
    #print(y_true.shape)
    errors = 0
    row_index = 0
    for row in p_y_x:
        biggest_prob = 0.0
        category_index = 0
        best_category = 0
        for prob_index in range(row.size):
            if biggest_prob <= row[prob_index]:
                biggest_prob = row[prob_index]
                best_category = category_index + 1
                #print(best_category)
            category_index += 1
        if best_category != y_true[row_index]:
            errors += 1
        row_index += 1
    #print(y_true.size)
    #print(errors/y_true.size)
    return errors/y_true.size



def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    """
    :param Xval: zbior danych walidacyjnych N1xD
    :param Xtrain: zbior danych treningowych N2xD
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartosci parametru k, ktore maja zostac sprawdzone
    :return: funkcja wykonuje selekcje modelu knn i zwraca krotke (best_error,best_k,errors), gdzie best_error to najnizszy
    osiagniety blad, best_k - k dla ktorego blad byl najnizszy, errors - lista wartosci bledow dla kolejnych k z k_values
    """
    #print(Xval.shape)
    #print(k_values)
    #print(yval)
    #print(ytrain)

    distance_matrix = hamming_distance(Xval, Xtrain)
    sorted_distance = sort_train_labels_knn(distance_matrix, ytrain)
    errors = []
    for k in k_values:
        #print(k)
        probability_matrix = p_y_x_knn(sorted_distance, k)
        error = classification_error(probability_matrix, yval)
        errors.append(error)
    #print(errors)
    min_error = min(errors)
    min_error_index = errors.index(min_error)
    return min_error, k_values[min_error_index], errors



def estimate_a_priori_nb(ytrain):
    """
    :param ytrain: etykiety dla dla danych treningowych 1xN
    :return: funkcja wyznacza rozklad a priori p(y) i zwraca p_y - wektor prawdopodobienstw a priori 1xM
    """
    categories_set = set(ytrain)
    categories_count = []
    for i in range(len(categories_set)):
        categories_count.append(np.count_nonzero(ytrain == i+1))
    return np.array(categories_count)/ytrain.shape[0]



def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    """
    :param Xtrain: dane treningowe NxD
    :param ytrain: etykiety klas dla danych treningowych 1xN
    :param a: parametr a rozkladu Beta
    :param b: parametr b rozkladu Beta
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(x|y) zakladajac, ze x przyjmuje wartosci binarne i ze elementy
    x sa niezalezne od siebie. Funkcja zwraca macierz p_x_y o wymiarach MxD.
    """
    x_decompresed = Xtrain.todense().astype(int)

    up_factor = a - 1.0
    down_factor = a + b - 2.0
    categories_set = set(ytrain)
    categories_count = []
    for i in range(len(categories_set)):
        categories_count.append(np.count_nonzero(ytrain == i+1))

    probabilities_matrix = np.zeros(shape=(len(categories_set), x_decompresed.shape[1]))

    for row_index in range(x_decompresed.shape[0]):
        prob_matrix_row_index = ytrain[row_index] -1
        for word_index in range(x_decompresed[row_index].size):
            if x_decompresed[row_index, word_index] == 1:
                probabilities_matrix[prob_matrix_row_index, word_index] += 1

    for row_index in range(probabilities_matrix.shape[0]):
        for column_index in range(probabilities_matrix.shape[1]):
            probabilities_matrix[row_index, column_index] = (probabilities_matrix[row_index, column_index] + up_factor)\
                                                            / (categories_count[row_index] + down_factor)
    return probabilities_matrix




def p_y_x_nb(p_y, p_x_1_y, X):
    """
    :param p_y: wektor prawdopodobienstw a priori o wymiarach 1xM
    :param p_x_1_y: rozklad prawdopodobienstw p(x=1|y) - macierz MxD
    :param X: dane dla ktorych beda wyznaczone prawdopodobienstwa, macierz NxD
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla kazdej z klas z wykorzystaniem klasyfikatora Naiwnego
    Bayesa. Funkcja zwraca macierz p_y_x o wymiarach NxM.
    """
    prob_for_classes = []

    x = X.toarray()
    prob_word_in_class = p_x_1_y
    prob_word_not_in_class = 1 - prob_word_in_class
    x_not = 1 - x

    for row_index in range(x.shape[0]):
        success = prob_word_in_class ** x[row_index]
        fail = prob_word_not_in_class ** x_not[row_index]
        product_matrix = success * fail
        numerator = np.prod(product_matrix, axis=1) * p_y
        denominator = sum(numerator)
        prob_for_classes.append(numerator/denominator)
    return prob_for_classes



def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):
    """
    :param Xtrain: zbior danych treningowych N2xD
    :param Xval: zbior danych walidacyjnych N1xD
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrow a do sprawdzenia
    :param b_values: lista parametrow b do sprawdzenia
    :return: funkcja wykonuje selekcje modelu Naive Bayes - wybiera najlepsze wartosci parametrow a i b. Funkcja zwraca
    krotke (error_best, best_a, best_b, errors) gdzie best_error to najnizszy
    osiagniety blad, best_a - a dla ktorego blad byl najnizszy, best_b - b dla ktorego blad byl najnizszy,
    errors - macierz wartosci bledow dla wszystkich par (a,b)
    """
    a_length = len(a_values)
    b_length = len(b_values)
    p_y = estimate_a_priori_nb(ytrain)
    errors = np.zeros((a_length, b_length))

    for a in range(0, a_length):
        for b in range(0, b_length):
            p_x_y_nb = estimate_p_x_y_nb(Xtrain, ytrain, a_values[a], b_values[b])
            p_y_x = p_y_x_nb(p_y, p_x_y_nb, Xval)
            errors[a, b] = classification_error(p_y_x, yval)

    lowest_error_index = np.argmin(errors)
    lowest_a_index = lowest_error_index // a_length
    lowest_b_index = lowest_error_index % b_length

    return errors[lowest_a_index, lowest_b_index], a_values[lowest_a_index], b_values[lowest_b_index], errors
























