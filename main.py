# =================================  TESTY  ===================================
# Testy do tego pliku zostały podzielone na dwie kategorie:
#
#  1. `..._invalid_input`:
#     - Sprawdzające poprawną obsługę nieprawidłowych danych wejściowych.
#
#  2. `..._correct_solution`:
#     - Weryfikujące poprawność wyników dla prawidłowych danych wejściowych.
# =============================================================================
import numpy as np
import numpy.polynomial.polynomial as nppoly


def roots_20(coef: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Funkcja wyznaczająca miejsca zerowe wielomianu funkcją
    nppoly.polyroots(), najpierw lekko zaburzając wejściowe współczynniki 
    wielomianu (N(0,1) * 1e-10).

    Args:
        coef (np.ndarray): Wektor współczynników wielomianu (n,).

    Returns:
        (tuple[np.ndarray, np. ndarray]):
            - Zaburzony wektor współczynników (n,),
            - Wektor miejsc zerowych (m,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca None.
    """
    if not isinstance(coef, np.ndarray):
        return None
    if coef.ndim != 1:
        return None
    if coef.size < 2:
        return None
    
    wspol = coef + (10**(-10)*np.random.random(len(coef)))
    zer = nppoly.polyroots(wspol)
    return wspol, zer

def frob_a(coef: np.ndarray) -> np.ndarray | None:
    """Funkcja służąca do wyznaczenia macierzy Frobeniusa na podstawie
    współczynników jej wielomianu charakterystycznego:
    w(x) = a_n*x^n + a_{n-1}*x^{n-1} + ... + a_2*x^2 + a_1*x + a_0

    Testy wymagają poniższej definicji macierzy Frobeniusa (implementacja dla 
    innych postaci nie jest zabroniona):
    F = [[       0,        1,        0,   ...,            0],
         [       0,        0,        1,   ...,            0],
         [       0,        0,        0,   ...,            0],
         [     ...,      ...,      ...,   ...,          ...],
         [-a_0/a_n, -a_1/a_n, -a_2/a_n,   ..., -a_{n-1}/a_n]]

    Args:
        coef (np.narray): Wektor współczynników wielomianu (n,).

    Returns:
        (np.ndarray): Macierz Frobeniusa o rozmiarze (n,n).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(coef, np.ndarray):
        return None
    if coef.ndim != 1 or coef.size < 2:
        return None

    coef = coef.astype(float)
    a_n = coef[0]
    if a_n == 0:
        return None

    n = coef.size - 1
    F = np.zeros((n, n))

    # jedynki nad diagonalą
    if n > 1:
        F[np.arange(n - 1), np.arange(1, n)] = 1.0


    last_row = -coef[1:] / a_n
    F[-1, :] = last_row[::-1]


    return F


def is_nonsingular(A: np.ndarray) -> bool | None:

    if not isinstance(A, np.ndarray):
        return None
    if A.ndim != 2:
        return None
    if A.shape[0] != A.shape[1]:
        return None

    try:
        det = np.linalg.det(A)
    except Exception:
        return None

    eps = np.finfo(float).eps
    return abs(det) > eps
