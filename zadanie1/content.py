# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np

from utils import polynomial


def mean_squared_error(x, y, w):
    """
    :param x: ciag wejsciowy Nx1
    :param y: ciag wyjsciowy Nx1
    :param w: parametry modelu (M+1)x1
    :return: blad sredniokwadratowy pomiedzy wyjsciami y
    oraz wyjsciami uzyskanymi z wielowamiu o parametrach w dla wejsc x
    """
    error = 0.0
    wartosciZModelu = polynomial(x, w)
    for i in range(y.size):
        error += (y[i][0] - wartosciZModelu[i][0])**2
    error /= y.size
    return error



def design_matrix(x_train, M):
    """
    :param x_train: ciag treningowy Nx1
    :param M: stopien wielomianu 0,1,2,...
    :return: funkcja wylicza Design Matrix Nx(M+1) dla wielomianu rzedu M
    """
    list = []
    #print(x_train)
    for i in range(x_train.size):
        rowList = []
        for j in range(M+1):
            #wypełniam wiersze macierzy - każdy kolejny element podniesiony jest do kolejnej potęgi
            rowList.append(x_train[i][0]**j)
        list.append(rowList)
    #print(list)
    #robie macierz numpajową z listy list
    return np.array(list)




def least_squares(x_train, y_train, M):
    """
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu, a err blad sredniokwadratowy
    dopasowania
    """
    designMatrix = design_matrix(x_train, M)
    # np.liang.inv zwraca macierz odwrotną
    w = np.linalg.inv(designMatrix.transpose() @ designMatrix) @ designMatrix.transpose() @ y_train
    return w, mean_squared_error(x_train, y_train, w)




def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    """
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu zgodnie z kryterium z regularyzacja l2,
    a err blad sredniokwadratowy dopasowania
    """
    designMatrix = design_matrix(x_train, M)
    mJednostkowa = regularization_lambda * np.eye(designMatrix.shape[1])
    # np.eye daje nam macierz jednostkową o zadanym kształcie
    w = np.linalg.inv((designMatrix.transpose() @ designMatrix) + mJednostkowa) @ designMatrix.transpose() @ y_train
    return w, mean_squared_error(x_train, y_train, w)


def model_selection(x_train, y_train, x_val, y_val, M_values):
    """
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M_values: tablica stopni wielomianu, ktore maja byc sprawdzone
    :return: funkcja zwraca krotke (w,train_err,val_err), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym, train_err i val_err to bledy na sredniokwadratowe na ciagach treningowym
    i walidacyjnym
    """
    models = []
    for i in M_values:
        val = least_squares(x_train, y_train, i)
        # val[0] to wagi/parametry wyznaczonego wielomianu do porównania
        validation_score = mean_squared_error(x_val, y_val, val[0])
        # dodanie kolejnego elementu do krotki - na jej koniec - operator przecinka ","
        val += validation_score,
        models.append(val)

    # znalezienie najlepszego modelu - o najmniejszym błędzie
    val = models[0]
    for i in models:
        if (i[2] < val[2]):
            val = i
    return val


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    """
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M: stopien wielomianu
    :param lambda_values: lista ze wartosciami roznych parametrow regularyzacji
    :return: funkcja zwraca krotke (w,train_err,val_err,regularization_lambda), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym. Wielomian dopasowany jest wg kryterium z regularyzacja. train_err i val_err to
    bledy na sredniokwadratowe na ciagach treningowym i walidacyjnym. regularization_lambda to najlepsza wartosc parametru regularyzacji
    """
    weights = []
    for i in lambda_values:
        # parametry w wyznaczone z danego stopnia wielomianu i lambdy razem z błędem - krotka
        weight_with_error = regularized_least_squares(x_train, y_train, M, i)
        # weight[0] to wagi wielomianu
        walidation_error = mean_squared_error(x_val, y_val, weight_with_error[0])
        # dodanie na koniec krotki błędu dla ciagu walidacyjnego i wartości parametru lambda dla ktorego zostal on policzony
        weights.append(weight_with_error + (walidation_error,) + (i,))
    best = weights[0]
    # szukanie najmiejszego błędu na ciągu walidacyjnym
    for i in weights:
        if (best[2] > i[2]):
            best = i
    return best

















