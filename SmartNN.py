import numpy as np
from tensorflow import keras
from numba import njit, jit
from time import perf_counter
import matplotlib.pyplot as plt

tr = perf_counter()

(InputTrain, IndexTrain), (InputTest, IndexTest) = keras.datasets.mnist.load_data()
InputTrain = InputTrain / 255
InpInf = InputTrain.reshape(60000, 784)
InputTest = InputTest / 255
InpTest = InputTest.reshape(10000, 784)

layers = [784, 512, 128, 32, 10]
with open('neurons.npy', 'rb') as n:
    neurons = np.load(n, allow_pickle=True)
with open('weights.npy', 'rb') as w:
    weights = np.load(w, allow_pickle=True)
with open('biases.npy', 'rb') as b:
    biases = np.load(b, allow_pickle=True)


@njit(nogil=True)
def sigmoid(p):
    return 1 / (1 + np.exp(-p))


@njit(nogil=True)
def derSigmoid(p):
    return p * (1 - p)


@jit(forceobj=True, nogil=True)
def feedforward(inp):
    neurons[0][0] = inp.copy()
    for v in range(1, len(layers)):
        neurons[v][0] = neurons[v - 1][0] @ weights[v - 1][0]
        neurons[v][0] += biases[v][0]
        neurons[v][0] = sigmoid(neurons[v][0])
    return neurons[len(layers) - 1][0]


@jit(forceobj=True, fastmath=True, nogil=True)
def nnTest():
    timer = perf_counter()
    right = 0
    print("Начало тестирования обучения")
    for bch in range(10000):
        trgDigit = IndexTest[bch]
        OutTest = feedforward(InpTest[bch])
        probDigit = np.argmax(OutTest)
        if trgDigit == probDigit:
            right += 1
    print("Конец тестирования. Точность нейросети:", right/100, "Время на анализ:", np.around(perf_counter() - timer, 2), "секунд")


def nnTest2():
    while True:
        dig = int(input("Число от 0 до 10 000, -1 для выхода: "))
        if dig == -1:
            break
        img = InputTest[dig]
        result = list(feedforward(InpTest[dig]))
        mx = max(result)
        print("Изображено число", result.index(mx), "с вероятностью", round(mx*100, 1))
        plt.imshow(img)
        plt.show()


nnTest()
plt.show()
nnTest2()
