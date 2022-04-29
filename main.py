from scipy.optimize import root,fsolve
import numpy as np
from matplotlib import pyplot as plt
from math import *



def first():

    def f(x, y, a, b, c, d, e, k):
        return a * cos(x) + b * sin(y) - c, d * cos(y) + e * sin(x) - k

    def matr_yakobi(v):
        matrix = [[]]

        w, h = 2, 2
        matrix = [[0 for x in range(w)] for y in range(h)]

        x = v[0]
        y = v[1]

        matrix[0][0] = -a * sin(x)
        matrix[0][1] = b * cos(y)

        matrix[1][0] = e * cos(x)
        matrix[1][1] = -d * sin(y)

        return matrix

    def find_T(x0):
        x0 = np.array(x0)
        x0 = x0.T
        return x0

    # подсчитываем обратную матрицу к матрице Якоби.
    def find_Yakobi_inverse(Y):
        Y = np.matrix(Y)
        Y = Y.I
        new_Y = Y.tolist()
        return new_Y

    # умножаем матрицу Якоби a на вектор b
    def matmult(a, b):
        a0 = a[0][0] * b[0] + a[0][1] * b[1]
        a1 = a[1][0] * b[0] + a[1][1] * b[1]
        return [a0, a1]

    # применяаем метод Ньютона для поиска новых значений x,y: здесь x0 -- это список, состоящий из x, y
    def iteration(x0, a, b, c, d, e, k):
        fx0 = f(x0[0], x0[1], a, b, c, d, e, k)
        Yakobi = matr_yakobi(x0)

        det = np.linalg.det(Yakobi)
        if det == 0.0:
            print('DET=0, ERROR')

        Yakobi_inverse = find_Yakobi_inverse(Yakobi)
        y1 = matmult(Yakobi_inverse, fx0)

        x1 = [0.0, 0.0]
        return [x0[0] - y1[0], x0[1] - y1[1]]

    a = 100
    b = 1.5
    c = -0.5
    d = -1.5
    e = 2
    k = 1

    a = int(input('Введите значение a:'))
    b = int(input('Введите значение b:'))
    c = float(input('Введите значение c:'))
    d = int(input('Введите значение d:'))
    e = int(input('Введите значение e:'))
    k = float(input('Введите значение k:'))
    aa = float(input('Введите погрешность:'))

    x0 = [0.5, 0.5]

    x = x0[0]
    y = x0[1]

    while abs(a * cos(x) + b * sin(y) - c) >= aa or abs(e * sin(x) + d * cos(y) - k) >= aa:
        x0 = iteration(x0, a, b, c, d, e, k)
        x = x0[0]
        y = x0[1]

    print('x=', x0[0], 'y=', x0[1])




def second():
    def f2(x):
        return 2 * np.sin(x) - x + 2

    x = np.linspace(-5, 5, num=100)
    y = f2(x)
    root1 = fsolve(f2, [1])  # Фактически, это корень системы уравнений, равный 0


    root2 = root(f2, [1])
    print(root1)
    print(root2)
    plt.plot(x, y, 'r')
    plt.show()

if __name__ == '__main__':
    pass
    # first() or second()