# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 3: Regresja logistyczna
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba, P. Dąbrowski
#  2019
# --------------------------------------------------------------------------

import numpy as np


def sigmoid(x):
    """
    Wylicz wartość funkcji sigmoidalnej dla punktów *x*.

    :param x: wektor wartości *x* do zaaplikowania funkcji sigmoidalnej Nx1
    :return: wektor wartości funkcji sigmoidalnej dla wartości *x* Nx1
    """
    return 1/(1+np.exp(-x))


def logistic_cost_function(w, x_train, y_train):
    """
    Wylicz wartość funkcji logistycznej oraz jej gradient po parametrach.

    :param w: wektor parametrów modelu Mx1
    :param x_train: zbiór danych treningowych NxM
    :param y_train: etykiety klas dla danych treningowych Nx1
    :return: krotka (log, grad), gdzie *log* to wartość funkcji logistycznej,
        a *grad* jej gradient po parametrach *w* Mx1
    """
    N = x_train.shape[0]
    sigmoid_matrix = sigmoid(x_train @ w)
    log = 0.0

    for i in range(N):
        log += y_train[i] * np.log(sigmoid_matrix[i]) + (1 - y_train[i]) * np.log(1 - sigmoid_matrix[i])
    log = log/N*(-1)
    gradient = 1/N*(np.transpose(x_train) @ (sigmoid_matrix - y_train))
    return log, gradient




def gradient_descent(obj_fun, w0, epochs, eta):
    """
    Dokonaj *epochs* aktualizacji parametrów modelu metodą algorytmu gradientu
    prostego, korzystając z kroku uczenia *eta* i zaczynając od parametrów *w0*.
    Wylicz wartość funkcji celu *obj_fun* w każdej iteracji. Wyznacz wartość
    parametrów modelu w ostatniej epoce.

    :param obj_fun: optymalizowana funkcja celu, przyjmująca jako argument
        wektor parametrów *w* [wywołanie *val, grad = obj_fun(w)*]
    :param w0: początkowy wektor parametrów *w* Mx1
    :param epochs: liczba epok algorytmu gradientu prostego
    :param eta: krok uczenia
    :return: krotka (w, log_values), gdzie *w* to znaleziony optymalny
        punkt *w*, a *log_values* to lista wartości funkcji celu w każdej
        epoce (lista o długości *epochs*)
    """
    log_values = np.zeros(epochs)
    w = w0.copy()
    for i in range(epochs):
        _, grad = obj_fun(w)
        delta_w = -grad
        w += eta*delta_w
        val, _ = obj_fun(w)
        log_values[i] = val
    return w, log_values




def stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch):
    """
    Dokonaj *epochs* aktualizacji parametrów modelu metodą stochastycznego
    algorytmu gradientu prostego, korzystając z kroku uczenia *eta*, paczek
    danych o rozmiarze *mini_batch* i zaczynając od parametrów *w0*. Wylicz
    wartość funkcji celu *obj_fun* w każdej iteracji. Wyznacz wartość parametrów
    modelu w ostatniej epoce.

    :param obj_fun: optymalizowana funkcja celu, przyjmująca jako argumenty
        wektor parametrów *w*, paczkę danych składających się z danych
        treningowych *x* i odpowiadających im etykiet *y*
        [wywołanie *val, grad = obj_fun(w, x, y)*]
    :param w0: początkowy wektor parametrów *w* Mx1
    :param epochs: liczba epok stochastycznego algorytmu gradientu prostego
    :param eta: krok uczenia
    :param mini_batch: rozmiar paczki danych / mini-batcha
    :return: krotka (w, log_values), gdzie *w* to znaleziony optymalny
        punkt *w*, a *log_values* to lista wartości funkcji celu dla całego
        zbioru treningowego w każdej epoce (lista o długości *epochs*)
    """
    w = w0.copy()
    M = (int)(x_train.shape[0] / mini_batch)
    log_values = np.zeros(epochs)

    for k in range(epochs):
        for mini_b_number in range(M):
            x_mini_batch = x_train[mini_b_number*mini_batch : mini_b_number*mini_batch + mini_batch]
            y_mini_batch = y_train[mini_b_number*mini_batch : mini_b_number*mini_batch + mini_batch]

            _, grad = obj_fun(w, x_mini_batch, y_mini_batch)
            delta_w = -grad
            w += delta_w * eta
        # dla całego zbioru danych a nie poszczególnych mini_batchów
        val, _ = obj_fun(w, x_train, y_train)
        log_values[k] = val
    return w, log_values

def regularized_logistic_cost_function(w, x_train, y_train, regularization_lambda):
    """
    Wylicz wartość funkcji logistycznej z regularyzacją l2 oraz jej gradient
    po parametrach.

    :param w: wektor parametrów modelu Mx1
    :param x_train: zbiór danych treningowych NxM
    :param y_train: etykiety klas dla danych treningowych Nx1
    :param regularization_lambda: parametr regularyzacji l2
    :return: krotka (log, grad), gdzie *log* to wartość funkcji logistycznej
        z regularyzacją l2, a *grad* jej gradient po parametrach *w* Mx1
    """
    logistic_without_reg, grad = logistic_cost_function(w, x_train, y_train)
    w_copy = w.copy()
    w_copy[0] = 0
    regularization = regularization_lambda/2 * (np.transpose(w_copy) @ w_copy)

    log = logistic_without_reg + regularization
    grad_with_reg = grad + regularization_lambda * w_copy
    return log, grad_with_reg

def prediction(x, w, theta):
    """
    Wylicz wartości predykowanych etykiet dla obserwacji *x*, korzystając
    z modelu o parametrach *w* i progu klasyfikacji *theta*.

    :param x: macierz obserwacji NxM
    :param w: wektor parametrów modelu Mx1
    :param theta: próg klasyfikacji z przedziału [0,1]
    :return: wektor predykowanych etykiet ze zbioru {0, 1} Nx1
    """
    #result = [ [1] if sigmoid(np.transpose(w) @ row) >= theta else [0] for row in x ]
    #print(result)
    result = np.zeros((x.shape[0],1))
    for i in range(x.shape[0]):
        if sigmoid(np.transpose(w) @ x[i]) >= theta:
            result[i] = 1
        else:
            result[i] = 0
    return result

def f_measure(y_true, y_pred):
    """
    Wylicz wartość miary F (F-measure) dla zadanych rzeczywistych etykiet
    *y_true* i odpowiadających im predykowanych etykiet *y_pred*.

    :param y_true: wektor rzeczywistych etykiet Nx1
    :param y_pred: wektor etykiet predykowanych przed model Nx1
    :return: wartość miary F (F-measure)
    """
    true_positives = np.sum(y_true * y_pred)
    model_mistakes = np.sum(np.logical_xor(y_pred, y_true))

    return 2*true_positives/(2*true_positives + model_mistakes)


def model_selection(x_train, y_train, x_val, y_val, w0, epochs, eta, mini_batch, lambdas, thetas):
    """
    Policz wartość miary F dla wszystkich kombinacji wartości regularyzacji
    *lambda* i progu klasyfikacji *theta. Wyznacz parametry *w* dla modelu
    z regularyzacją l2, który najlepiej generalizuje dane, tj. daje najmniejszy
    błąd na ciągu walidacyjnym.

    :param x_train: zbiór danych treningowych NxM
    :param y_train: etykiety klas dla danych treningowych Nx1
    :param x_val: zbiór danych walidacyjnych NxM
    :param y_val: etykiety klas dla danych walidacyjnych Nx1
    :param w0: początkowy wektor parametrów *w* Mx1
    :param epochs: liczba epok stochastycznego algorytmu gradientu prostego
    :param eta: krok uczenia
    :param mini_batch: rozmiar paczki danych / mini-batcha
    :param lambdas: lista wartości parametru regularyzacji l2 *lambda*,
        które mają być sprawdzone
    :param thetas: lista wartości progów klasyfikacji *theta*,
        które mają być sprawdzone
    :return: krotka (regularization_lambda, theta, w, F), gdzie
        *regularization_lambda* to wartość regularyzacji *lambda* dla
        najlepszego modelu, *theta* to najlepszy próg klasyfikacji,
        *w* to parametry najlepszego modelu, a *F* to macierz wartości miary F
        dla wszystkich par *(lambda, theta)* #lambda x #theta
    """

    F_matrix = np.zeros((len(lambdas), len(thetas)))
    best_f = 0
    best_w = 0
    best_theta = 0
    best_lambda = 0
    for lambd in range(len(lambdas)):
        obj_fun = lambda w, x_tr, y_tr: regularized_logistic_cost_function(w, x_tr, y_tr, lambdas[lambd])
        current_w, _ = stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch)
        for theta in range(len(thetas)):
            f = f_measure(y_val, prediction(x_val, current_w, thetas[theta]))
            F_matrix[lambd, theta] = f
            if f > best_f:
                best_f = f
                best_theta = thetas[theta]
                best_lambda = lambdas[lambd]
                best_w = current_w
    return best_lambda, best_theta, best_w, F_matrix








