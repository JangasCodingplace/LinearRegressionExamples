from sympy import Sum, symbols, Indexed, lambdify
from sympy import expand, simplify, diff, solve, Eq
import numpy as np

def get_sum(min_limit, max_limit, reg):
    x, i = symbols('x i')
    return Sum(Indexed('x',i),(i,min_limit, max_limit)).doit()

def get_best_configuration(regr, output):
    X = symbols(' '.join(['x_{}'.format(i) for i in range(len(regr.coef_))]))
    features = np.array(X)

    x = symbols('x')
    sum_RAW = get_sum(1,len(regr.coef_)-1, regr)
    sum_function_RAW = lambdify(x, sum_RAW)

    # einfache Summe bilden & substiuieren
    basic_factors = np.array([x**2 for x in features])
    abs_sum_function = sum_function_RAW(basic_factors)

    # Summe mit faktoren bilden & c-Werte einsetzen & substituieren
    coeff_factors = np.array([regr.coef_[i] * x for i,x in enumerate(features)])
    coeff_sum_function = sum_function_RAW(coeff_factors)

    # Ausklammern
    y = output - regr.intercept_
    expand_x_1 = expand((y - coeff_sum_function)**2)

    # c eingesetzt
    c_1 = symbols('c_1')
    c_1h_m1 = 1 / c_1**2
    c_1h_m1 = c_1h_m1.subs(c_1, regr.coef_[0])

    # x_1 Faktor
    x_1_subst = c_1h_m1 * expand_x_1

    # Vereinfachen
    rh2 = simplify(x_1_subst + abs_sum_function)

    # Differentiale
    differentiale = np.array([diff(rh2, 'x_1'),diff(rh2, 'x_2')])

    # LGS Loesen
    solved = solve(differentiale)

    # x_1 bestimmen
    x_1_value = x_1_subst.subs([(key, value**0.5) for key, value in solved.items()])

    return [x_1_value**0.5] + [value**0.5 for key, value in solved.items()]
