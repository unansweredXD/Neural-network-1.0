from tkinter import *
import numpy as np
import random
import pylab
import math
import time
from matplotlib import mlab
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

def displayVectors(data, classes, amountOfVectors, type, weights):
    im = Image.new('RGB', (1200, 1000), (170, 170, 170))
    draw = ImageDraw.Draw(im)
    fnt = ImageFont.truetype('arial.ttf', 14)
    fnt2 = ImageFont.truetype('arial.ttf', 14)
    if amountBitOfImage > 300 and amountBitOfImage < 1000:
        scale = 2
    elif amountBitOfImage > 1000:
        scale = 0.5
    else:
        scale = 8
    yShift = 5
    xShift = 5
    yL = yShift
    xL = xShift
    if type == 0:
        for i in range(amountOfVectors):
            displayOneVector(data, classes, weights, i, xL, yL, scale, draw, fnt, fnt2)
            xL += scale * matrixSizeX + xShift
            if  xL + scale * matrixSizeX + xShift > 1200:
                yL += scale * matrixSizeY + yShift + 35
                if yL + scale * matrixSizeY + yShift + 35 > 1000:
                    break
                xL = xShift
    else:
        for i in range(amountOfVectors):
            displayOneVectorHalfTone(data, classes, weights, i, xL, yL, scale, draw, fnt, fnt2)
            xL += scale * matrixSizeX + xShift
            if  xL + scale * matrixSizeX + xShift > 1200:
                yL += scale * matrixSizeY + yShift + 35
                if yL + scale * matrixSizeY + yShift + 35 > 1000:
                    break
                xL = xShift
    im.show()

def displayOneVector(data, classes, weights, it, xL, yL, scale, draw, fnt, fnt2):
    currentVec = it
    currentX = xL
    currentY = yL
    currentScale = scale
    for i in range(matrixSizeY):
        for j in range(matrixSizeX):
            z = data[currentVec][i * matrixSizeX + j]
            if z == 1:
                draw.rectangle(((currentX, currentY), (currentX + scale, currentY + scale)), fill = "black")
            else:
                draw.rectangle(((currentX, currentY), (currentX + scale, currentY + scale)), fill = "white")
            currentX += scale
        currentY += scale
        currentX = xL
    draw.text((currentX, currentY + 5), str(classes[currentVec]), fill = (255, 255, 0), font = fnt)
    wMax = max(weights[currentVec])
    numMax = 0
    for i in range(outputClasses):
        if weights[currentVec][i] == wMax:
            numMax = i + 1
    draw.text((currentX, currentY + 20), str(numMax), fill = (255, 0, 0), font = fnt2)

def displayOneVectorHalfTone(data, classes, weights, it, xL, yL, scale, draw, fnt, fnt2):
    currentVec = it
    currentX = xL
    currentY = yL
    for i in range(matrixSizeY):
        for j in range(matrixSizeX):
            z = data[currentVec][i * matrixSizeX + j]
            draw.rectangle(((currentX, currentY), (currentX + scale, currentY + scale)), fill = (z, z, z))
            currentX += scale
        currentY += scale
        currentX = xL
    draw.text((currentX, currentY + 5), str(classes[currentVec]), fill = (255, 255, 0), font = fnt)
    wMax = max(weights[currentVec])
    numMax = 0
    for i in range(outputClasses):
        if weights[currentVec][i] == wMax:
            numMax = i + 1
    draw.text((currentX, currentY + 20), str(numMax), fill = (255, 0, 0), font = fnt2)

def readLearningFile(fileName):
    global matrixSizeX
    global matrixSizeY
    global amountOfLearningImages
    global dataForLearning
    global classesForLearning
    global typeOfImages
    global outputClasses
    dataForLearning = []
    classesForLearning = []

    try:
        f = open(fileName, 'r')
    except FileNotFoundError:
        lbError.config(text = "Файл с таким именем не найден")
        return 0

    s = f.readline().split(' ')
    typeOfImages = int(s[0])
    matrixSizeX = int(s[1])
    matrixSizeY = int(s[2])
    outputClasses = int(s[3])
    amountOfLearningImages = int(s[4])

    for k in range(amountOfLearningImages):
        vector = []
        for i in range(matrixSizeY):
            s = f.readline().split('\n')
            s = s[0].split(' ')
            for j in range(matrixSizeX):
                vector.append(int(s[j]))
        dataForLearning.append(vector)
        s = f.readline()
        classesForLearning.append(int(s))
    f.close()
    return 1

def fillWeightArr(amountOfStr, amountOfCol):
    weightsOfSyn = []
    for i in range(amountOfStr):
        randomWeights = []
        for j in range(amountOfCol):
            randomWeights.append(random.uniform(-1, 1))
        weightsOfSyn.append(randomWeights)
    return weightsOfSyn

def sigmoid(x):
    if (x < -700):
        y = 0
    else:
        y = 1.0/(1 + math.exp(-x))
    return y

def forwardPath(inp):
    x = np.zeros((3, amountBitOfImage), dtype = np.float64)
    y = np.zeros((3, amountBitOfImage), dtype = np.float64)
    for i in range(len(inp)):
        x[0][i] = inp[i]
        y[0][i] = inp[i]
    for i in range(amountOfNeurons):
        for j in range (amountBitOfImage):
            x[1][i] += y[0][j] * weightsOfSynapses[0][j][i]
        y[1][i] = sigmoid(x[1][i])

    for i in range(outputClasses):
        for j in range (amountOfNeurons):
            x[2][i] += y[1][j] * weightsOfSynapses[1][j][i]
        y[2][i] = sigmoid(x[2][i])
    return x, y

def calculateMistake(currentClass):
    global d
    d = []
    for j in range(outputClasses):
        if j == currentClass - 1:
            d.append(1)
        else:
            d.append(0)
    Ez = sum((y[len(y)-1][j] - d[j])**2 for j in range(outputClasses))/2
    return Ez

def diffSigm(x):
    y = sigmoid(x)
    return y * (1 - y)

def backward():

    EY = np.zeros((2, amountOfNeurons), dtype = np.float64)
    EX = np.zeros((2, amountOfNeurons), dtype = np.float64)
    EW = np.zeros((2, amountBitOfImage, amountOfNeurons), dtype = np.float64)

    for i in range(outputClasses):
        EY[1][i] = y[2][i] - d[i]
        EX[1][i] = EY[1][i] * diffSigm(y[2][i])

    for i in range(outputClasses):
        for j in range(amountOfNeurons):
            EW[1][j][i] = y[1][j] * EX[1][i]

    for i in range(amountOfNeurons):
        for j in range(outputClasses):
            EY[0][i] += EX[1][j] * weightsOfSynapses[1][i][j]
        EX[0][i] = EY[0][i] * diffSigm(y[1][i])

    for i in range(amountBitOfImage):
        for j in range(amountOfNeurons):
            EW[0][i][j] = EX[0][j] * y[0][i]

    for i in range(2):
        for j in range(len(weightsOfSynapses[i])):
            for k in range(len(weightsOfSynapses[i][j])):
                weightsOfSynapses[i][j][k] -= delta * EW[i][j][k]

def learning():
    startTime = time.perf_counter()
    fileExsist = readLearningFile(inputLearning)
    if fileExsist == 0:
        return
    global weightsOfSynapses
    global delta
    global isLearningStarted
    global amountBitOfImage
    global amountOfNeurons
    global x
    global y
    
    amountBitOfImage = matrixSizeX * matrixSizeY

    try:
        amountOfNeurons = int(enAmountOfNeurons.get())
        amountOfEras = int(enAmountOfEras.get())
        maxMistake = float(enMaxMistake.get())
        delta = float(enDelta.get())
    except ValueError:
        lbError.config(text = "Введено некорректное значение, повторите попытку")
        return 0
    
    weightsOfSynapses = []

    weightsOfSynapses.append(fillWeightArr(amountBitOfImage, amountOfNeurons))
    weightsOfSynapses.append(fillWeightArr(amountOfNeurons, outputClasses))

    numberOfEra = 0
    E = []
    E.append(1)

    while(numberOfEra < amountOfEras and E[numberOfEra] > maxMistake):
        Ei = 0 
        for i in range(amountOfLearningImages):
            inp = []
            for j in range(amountBitOfImage):
                inp.append(dataForLearning[i][j])
            x, y = forwardPath(inp)
            Ez = calculateMistake(classesForLearning[i])
            Ei += Ez
            backward()

        E.append(Ei/amountOfLearningImages)
        numberOfEra += 1
    xlist = [x for x in range(1, len(E))]
    ylist = [E[x] for x in xlist]
    finishTime = time.perf_counter()
    successStr = f"{finishTime - startTime:0.3f}"
    lbError.config(text = 'Обучение прошло успешно.')
    lbSuccess.config(text = 'Потраченное время: ' + successStr + ' сек.')
    isLearningStarted = True
    pylab.plot (xlist, ylist, "m")
    pylab.show()
    
def readTestFile(fileName):
    global matrixSizeX
    global matrixSizeY
    global amountOfImagesForRecog
    global dataForRecognizing
    global typeOfImages
    global outputClasses
    global testClasses
    dataForRecognizing = []
    testClasses = []

    try:
        f = open(fileName, 'r')
    except FileNotFoundError:
        lbError.config(text = "Файл с таким именем не найден")
        return 0

    s = f.readline().split(' ')
    typeOfImages = int(s[0])
    matrixSizeX = int(s[1])
    matrixSizeY = int(s[2])
    outputClasses = int(s[3])
    amountOfImagesForRecog = int(s[4])

    for k in range(amountOfImagesForRecog):
        vector = []
        for i in range(matrixSizeY):
            s = f.readline().split('\n')
            s = s[0].split(' ')
            for j in range(matrixSizeX):
                vector.append(int(s[j]))
        dataForRecognizing.append(vector)
        s = f.readline()
        testClasses.append(int(s))
    f.close()
    return 1

def recognition():
    try:
        if not isLearningStarted:
            return 0
    except BaseException:
        lbError.config(text = 'Сначала необходимо произвести обучение')
        return 0
    fileExsist = readTestFile(inputTest)
    if fileExsist == 0:
        return
    Y = []
    for i in range(amountOfImagesForRecog):
        inp = []
        for j in range(matrixSizeX * matrixSizeY):
            inp.append(dataForRecognizing[i][j])
        xRec, yRec = forwardPath(inp)
        Y.append(yRec[len(yRec)-1])

    lbSuccess.config(text = '')
    lbError.config(text = 'Распознавание прошло успешно')

    displayVectors(dataForRecognizing, testClasses, amountOfImagesForRecog, typeOfImages, Y)

inputLearning = "D:/grey_faces_5.txt"
inputTest = "D:/grey_faces_5.txt"

mainForm = Tk()
mainForm.title("Нейросеть")
mainForm.geometry("400x450")
lbAmountOfNeurons = Label(mainForm, text = "Число нейронов", anchor = W)
lbAmountOfNeurons.place(x = 10, y = 10, width = 100, height = 25)
enAmountOfNeurons = Entry(mainForm)
enAmountOfNeurons.place(x = 120, y = 10, width = 80, height = 25)
lbAmountOfEras = Label(mainForm, text = "Число эпох", anchor = W)
lbAmountOfEras.place(x = 10, y = 60, width = 100, height = 25)
enAmountOfEras = Entry(mainForm)
enAmountOfEras.place(x = 120, y = 60, width = 80, height = 25)
lbMaxMistake = Label(mainForm, text = "Макс. ошибка", anchor = W)
lbMaxMistake.place(x = 10, y = 110, width = 100, height = 25)
enMaxMistake = Entry(mainForm)
enMaxMistake.place(x = 120, y = 110, width = 80, height = 25)
lbDelta = Label(mainForm, text = "Дельта", anchor = W)
lbDelta.place(x = 10, y = 160, width = 100, height = 25)
enDelta = Entry(mainForm)
enDelta.place(x = 120, y = 160, width = 80, height = 25)
btLearning = Button(mainForm, text = "Обучение", command = learning)
btLearning.place(x = 83, y = 210, width = 100)
btRecognition = Button(mainForm, text = "Распознавание", command = recognition)
btRecognition.place(x = 217, y = 210, width = 100)
lbCorrectRecog = Label(mainForm, text = "", anchor = W)
lbCorrectRecog.place(x = 10, y = 300, width = 200, height = 25)
lbError = Label(mainForm, text = "", anchor = W)
lbError.place(x = 10, y = 340, width = 350, height = 25)
lbSuccess = Label(mainForm, text = "", anchor = W)
lbSuccess.place(x = 10, y = 380, width = 350, height = 25) 
mainForm.mainloop()
