import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from numba import njit, jit
from time import perf_counter

tr = perf_counter()

(InputTrain, IndexTrain), (InputTest, IndexTest) = keras.datasets.mnist.load_data()
InputTrain = InputTrain / 255
InpInf = InputTrain.reshape(60000, 784)
InputTest = InputTest / 255
InpTest = InputTest.reshape(10000, 784)

layers = [784, 512, 128, 32, 10]
learnRate = 0.01
epochs = 150
batchSize = 250


@njit(nogil=True)
def rd(i):
    return 2*np.random.random(i)-1


@njit(nogil=True)
def rdw(i, j):
    return 2*np.random.rand(i, j)-1


@njit(nogil=True)
def sigmoid(p):
    return 1 / (1 + np.exp(-p))


@njit(nogil=True)
def derSigmoid(p):
    return p * (1 - p)


neurons = [
    [rd(784)],
    [rd(layers[1])],
    [rd(layers[2])],
    [rd(layers[3])],
    [rd(10)]]
biases = [
    [rd(784)],
    [rd(layers[1])],
    [rd(layers[2])],
    [rd(layers[3])],
    [rd(10)]]
weights = [
    [rdw(784, layers[1])],
    [rdw(layers[1], layers[2])],
    [rdw(layers[2], layers[3])],
    [rdw(layers[3], 10)]]

print("Заполнены массивы смещений и весов", np.round(perf_counter()-tr, 2))


@jit(forceobj=True, nogil=True)
def feedforward(inp):
    neurons[0][0] = inp.copy()
    for v in range(1, len(layers)):
        neurons[v][0] = neurons[v - 1][0] @ weights[v - 1][0]
        neurons[v][0] += biases[v][0]
        neurons[v][0] = sigmoid(neurons[v][0])
    return neurons[len(layers) - 1][0]


@jit(forceobj=True, nogil=True)
def backpropagation(targets):
    error = targets - neurons[len(layers) - 1][0]
    for v in range(len(layers) - 2, -1, -1):
        gradients = error * derSigmoid(neurons[v + 1][0])
        gradients *= learnRate
        deltas = np.outer(gradients, neurons[v][0])
        error = weights[v][0] @ error
        deltas = np.rot90(deltas)
        deltas = np.flip(deltas, 0)
        weights[v][0] = weights[v][0] + deltas
        biases[v + 1][0] += gradients


@jit(forceobj=True, nogil=True)
def nn():
    for ep in range(epochs):
        timer = perf_counter()
        right = 0
        for bch in range(batchSize):
            rndImage = np.random.randint(0, 60000)
            trgDigit = IndexTrain[rndImage]
            target = np.zeros(10)
            target[trgDigit] = 1
            OutInf = feedforward(InpInf[rndImage])
            probDigit = np.argmax(OutInf)
            if trgDigit == probDigit:
                right += 1
            backpropagation(target)
        print("Эпоха:", ep, "Верных ответов", right, "/", batchSize, '\n', "Время на эпоху:", np.round(perf_counter() - timer, 2))
    print("конец, время затраченное на обучение", np.round((perf_counter() - tr)/60, 2), "минут")


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


nn()
# print(weights[3][0])
# c = 0
# with open("weight3.json", "w") as f:
#     for line in weights[3][0]:
#         lineNorm = [i for i in weights[3][0][c]]
#         json.dump(line, f)
#         c += 1
#        f.write('\n')
# with open("weight3.json", 'r') as f2:
#     wed = json.load(f2)
# neuro = [i for i in neurons[4][0]]
# with open('neurons4.json', 'w') as n:
#     json.dump(neuro, n)
# with open('neurons4.json', 'r') as l:
#     lst = json.load(l)
with open('neurons.npy', 'wb') as nt:
    np.save(nt, neurons)
with open('weights.npy', 'wb') as wt:
    np.save(wt, weights)
with open('biases.npy', 'wb') as bt:
    np.save(bt, biases)
nnTest()
plt.show()
nnTest2()
