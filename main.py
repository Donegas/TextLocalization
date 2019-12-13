# -*- coding: utf-8 -*-
"""
Thaís Donega
Localização de texto em Pôster
"""
from PIL import Image
import pytesseract as pyt
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from pythonRLSA import rlsa
import PIL.Image
import pillowfight

#Para que o Tesseract Funcione
pyt.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

# Le Imagem RGB e PB. Recebe Caminho e retorna 2 matrizes (1 para RGB ImgRGB e outra Cinza ImgPB)
def LeImg(Img):
    ImgRGB = cv2.imread(Img)
    ImgPB = cv2.cvtColor(ImgRGB, cv2.COLOR_BGR2GRAY)
    return ImgRGB, ImgPB

#Filtro Mediana para suavizar ruidos - melhor pq mantem as bordas agudas / nitidez das bordas
#Recebe uma imagem e devolve a mesma imagem com a suavização
def Mediana(Img):
    Img = cv2.medianBlur(Img, 3)
    return Img

#Detector de Bordas (Prewitt, Sobel e Canny)

#Prewitt: Aplica Máscaras H e V na Imagem de entrada e as retorna
def Prewitt(Img):
    # Delimita áreas de variação abruptas de intensidade
    # Aplica Filtros derivativos de Prewitt
    PrewittH = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    PrewittV = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    ImgPrewittH = cv2.filter2D(Img, -1, PrewittH)
    ImgPrewittV = cv2.filter2D(Img, -1, PrewittV)
    return ImgPrewittH, ImgPrewittV

#Prewitt: Aplica técnica de sobel do cv2 para x e y na Imagem de entrada e as retorna
def Sobel(Img):
    print("Aplicando Filros de Sobel para X e Y - Magnitude do Gradiente (43)")
    SobelX = cv2.Sobel(Img, cv2.CV_64F, 1, 0)
    SobelY = cv2.Sobel(Img, cv2.CV_64F, 0, 1)
    SobelX = np.uint8(abs(SobelX))
    SobelY = np.uint8(abs(SobelY))
    return SobelX, SobelY

def Canny(Img, Mag, sigma=0.3):
    print("Aplicando Detecção de Bordas Canny (Suavização com Gauss, Filtro de Sobel, Supressão de Não Máximos e Treshold (51)")
    v = np.mean(Mag)
    limiar1 = 150 #int(max(0, (1.0 - sigma) * v))
    limiar2 = 200 #int(min(255, (1.0 + sigma) * v))
    # Aplica a detecção de bordas canny do openv
    Canny = cv2.Canny(Img, limiar1, limiar2)
    return Canny

def MagStrength(ImgX, ImgY):
    print("Cálculo da força da Magnitude do Gradiente através da 2ª derivada dos filtros aplicados")
    ImgMag = np.zeros([ImgX.shape[0], ImgX.shape[1]])
    for x in range(ImgX.shape[0]):
        for y in range(ImgX.shape[1]):
            ImgMag[x, y] = np.sqrt(ImgX[x, y]**2 + ImgY[x, y]**2)
    return ImgMag

#Calcula direção do gradiente, considerando Bordas de X e Y resultantes das detecção de bordas escolhida #GTeta(i,j) = arctan(Gy(i,j)/Gx(i,j))
def Teta(ImgX, ImgY):
    print("Cálculo da direção do Gradiente em Graus (0 a 90)")
    ImgTeta = np.zeros([ImgX.shape[0], ImgX.shape[1]])
    for x in range(ImgX.shape[0]):
        for y in range(ImgX.shape[1]):
            if ImgX[x,y] != 0:
                ImgTeta[x, y] = np.arctan(ImgY[x,y] / ImgX[x,y])
            else:
                ImgTeta[x, y] = 0.0
            ImgTeta[x, y] = ImgTeta[x, y] * (180/np.pi)
    return ImgTeta

def MostraImg(texto, Img):
    cv2.imshow(texto, Img)
    cv2.waitKey(0)

def ConnectedComponents(Img):
    ImgCC = cv2.threshold(Img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    ret, ImgCC = cv2.connectedComponents(ImgCC)
    MakeVisible(ImgCC)
    return ImgCC

def ConnectedComponentsStats(Img):
    print("Cálculo dos Componentes Conectados com suas estatísticas possíveis (Top, Left, Height e Width + Pixels de borda/Área (151)")
    ImgCC = cv2.threshold(Img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    nlabels, ImgCC, stats, centroids = cv2.connectedComponentsWithStats(ImgCC)
    # stats[0], centroids[0] are for the background label. ignore
    #lblareas = stats[1:, cv2.CC_STAT_AREA] #Recebe a informação de área de cada componente (exceto fundo)
    #ave = np.average(centroids[1:], axis=0, weights=lblareas) #tupla de médias dos centroids com os pesos das áreas não sei pra que serve
    ImgCC = ImgCC.astype(np.uint8)
    #MostraImg("ConnectedComponentsStats em RGB", MakeVisible(ImgCC))
    #imax = max(enumerate(lblareas), key=(lambda x: x[1]))[0] +
    print("Total de Componentes: ", nlabels)
    return ImgCC, stats, centroids, nlabels

def CriaBoundingBoxes(stats, ImgRGB):
    print("Desenha os Bounding Boxes para cada Componente Conectado (163)")
    ImgRGBCopy = ImgRGB.copy()
    ImgBB = np.zeros([ImgRGB.shape[0], ImgRGB.shape[1]])
    for c in stats[1:, :4]:
        P1 = (c[cv2.CC_STAT_LEFT], c[cv2.CC_STAT_TOP])
        P2 = (c[cv2.CC_STAT_LEFT] + c[cv2.CC_STAT_WIDTH], (c[cv2.CC_STAT_TOP] + c[cv2.CC_STAT_HEIGHT]))

        # desenha retângulos em torno do componente
        cv2.rectangle(ImgRGBCopy, P1, P2, (0, 255, 0), thickness = 2) #Apenas para Visualização
        cv2.rectangle(ImgBB, P1, P2, (255), thickness=2) #Contém apenas os retângulos

    #MostraImg("ImgRGB", ImgRGBCopy) #Apenas para Vizualização: Img Colorida com os retângulos em verde
    print("Total de Bounding Boxes: ", stats.shape[0])
    return ImgRGBCopy

def MakeVisible(Img):
    # Map component labels to hue val
    label_hue = np.uint8(179 * Img / np.max(Img))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    # set bg label to black
    labeled_img[label_hue == 0] = 0
    cv2.imwrite(str(path+ imagem + " - 0ComponentesConectados" + extensao), labeled_img)
    return labeled_img

def BBAreasSelection(stats, ImgRGB):
    print("Seleção para inclusão ou exclusão de cada Componente Conectado de acordo com a Área do Bounding Box (190)")
    #Area minima: cada BB tem que ter pelo menos 20 pixels
    NovoStats = []; NovoStats.append(stats[0])
    AlturaImg = stats[0,3]; LarguraImg = stats[0,2]; AreaImg = AlturaImg*LarguraImg; MediaArea = np.mean(stats[1:, cv2.CC_STAT_HEIGHT] * stats[1:, cv2.CC_STAT_WIDTH])
    AlturaMedia = np.mean(stats[1:, cv2.CC_STAT_HEIGHT]); LarguraMedia = np.mean(stats[1:, cv2.CC_STAT_WIDTH])
    for i, CC in enumerate(stats[1:], start=1):
        Flag1 = False; Flag2 = False; Flag3 = False; Flag4 = False; Flag5 = False;
        AreaCC = CC[cv2.CC_STAT_HEIGHT] * CC[cv2.CC_STAT_WIDTH]
        DensityCC = CC[cv2.CC_STAT_AREA]; AspectRatio = CC[cv2.CC_STAT_WIDTH]/CC[cv2.CC_STAT_HEIGHT] ; Extent = AreaCC/DensityCC
        if DensityCC > 15 or AreaCC > 30: #Area do BB #Desidade dos pixels BB > 10 pixels E
            Flag1 = True
        if AreaCC < AreaImg*0.4: #Area Total BB < 40% da Img
            Flag2 = True
        if CC[cv2.CC_STAT_HEIGHT] <= AlturaImg*0.6 and CC[cv2.CC_STAT_WIDTH] <= LarguraImg*0.6: #Altura e largura do BB < 60% da altura e largura da Img
            Flag3 = True
        if CC[cv2.CC_STAT_HEIGHT] < 200 and CC[cv2.CC_STAT_WIDTH] < 150: #Tamanho máximo do BB 200H x 150W
            Flag4 = True
        #if AspectRatio > 0.2 and AspectRatio < 10.0: #Proporção de 0.5 a 10, priorizando retângulos "em pé"
        #    Flag5 = True
        if Flag1 == True and Flag2 == True and Flag3 == True and Flag4 == True:
            NovoStats.append(CC)
        #print("CC, Linha, Coluna, Altura, Largura, Extent, Density, Area, Ratio: \n",
        #      i, CC[cv2.CC_STAT_TOP], CC[cv2.CC_STAT_LEFT], CC[cv2.CC_STAT_HEIGHT], CC[cv2.CC_STAT_WIDTH],
        #      Extent, DensityCC, AreaCC, AspectRatio)
    NovoStats = np.asarray(NovoStats)
    ImgRGBCopy = CriaBoundingBoxes(NovoStats, ImgRGB)
    return NovoStats, ImgRGBCopy

def BBGradientSelection(stats, ImgRGB, ImgMag, ImgTeta, ImgCC):
    print("Seleção para inclusão ou exclusão de cada Componente Conectado de acordo com as informações da Magnitude e Direção do Gradiente (207)")
    NovoStats = []; MediaMag = 0;
    TreshAngle = 85 #De acordo com Liu
    NovoStats.append(stats[0])
    for i, CC in enumerate(stats[1:], start=1):
        Flag1 = False; Flag2 = False; Flag3 = False
        XMin = CC[cv2.CC_STAT_TOP]; XMax = CC[cv2.CC_STAT_TOP] + CC[cv2.CC_STAT_HEIGHT] #I inicial e final do BB
        YMin = CC[cv2.CC_STAT_LEFT]; YMax = CC[cv2.CC_STAT_LEFT] + CC[cv2.CC_STAT_WIDTH] #J inicial e final do BB
        TreshMag = LimiarM(ImgMag, ImgTeta, XMin, XMax, YMin, YMax)
        #Componente = np.min(ImgCC[XMin:XMax, YMin:YMax][np.nonzero(ImgCC[XMin:XMax, YMin:YMax])])
        for x in range (XMin, XMax):
            for y in range (YMin, YMax):
                #Media da Magnitude do gradiente dos pixels de contorno
                if ImgCC[x, y] != 0 and ImgMag[x, y] > 0: #Se for um pixel de borda e sua magnitude for > 0
                    MediaMag = (MediaMag + ImgMag[x, y])/CC[cv2.CC_STAT_AREA]
        #if MediaMag < TreshMag:
        #    Flag1 = True
        #Variação do Teta
        dif = np.max(ImgTeta[XMin:XMax, YMin:YMax]) - np.min(ImgTeta[XMin:XMax, YMin:YMax])
        if dif > TreshAngle:
            Flag2 = True
        #Count dos pixels de borda
        if CC[cv2.CC_STAT_AREA] > max(2 * CC[cv2.CC_STAT_HEIGHT], 2 * CC[cv2.CC_STAT_WIDTH]):
            Flag3 = True
        if Flag2 == True and Flag3 == True:
            NovoStats.append(CC)
        print("Componente: ", i, "; X: ", XMin, "; Y: ", YMin, "-> TresholdMag: ", TreshMag, "; MediaMag: ", MediaMag, "; TreshAngle: ", TreshAngle,
              "; DiferençaAngulo", dif, "; TreshDensity: ", max(2 * CC[cv2.CC_STAT_HEIGHT], 2 * CC[cv2.CC_STAT_WIDTH]),
              "; Densidade: ", CC[cv2.CC_STAT_AREA])
        print("Flag1 = ", Flag1, "; Flag2 = ", Flag2, "; Flag3 = ", Flag3)
    NovoStats = np.asarray(NovoStats)
    ImgRGBCopy = CriaBoundingBoxes(NovoStats, ImgRGB)
    return NovoStats, ImgRGBCopy

def LimiarM(ImgMag, ImgTeta, XMin, XMax, YMin, YMax):
    sum1 = 0; sum2 = 1
    if YMax == ImgMag.shape[1]:
        YMax = YMax-1
    if XMax == ImgMag.shape[0]:
        XMax = XMax-1
    for x in range (XMin, XMax):
        for y in range (YMin, YMax):
            MagPI = ImgMag[x, y]
            TetaPI = ImgTeta[x, y]
            if (TetaPI > 337.5 or TetaPI < 22.5) or (TetaPI > 157.5 and TetaPI < 202.5):  # Angulos próximos a 0 ou próximos a 180 - direita e esquerda
                Viz1 = ImgMag[x, y + 1]
                Viz2 = ImgMag[x, y - 1]
            elif (TetaPI > 22.5 and TetaPI < 67.5) or (TetaPI > 202.5 and TetaPI < 247.5):  # Angulos próximos a 45 ou próximos a 225 - diagonal superior direita e diagonal inferior esquerda
                Viz1 = ImgMag[x - 1, y + 1]
                Viz2 = ImgMag[x + 1, y - 1]
            elif (TetaPI > 67.5 and TetaPI < 112.5) or (TetaPI > 247.5 and TetaPI < 292.5):  # Angulos próximos a 90 ou próximos a 270 - Acima e Abaixo
                Viz1 = ImgMag[x - 1, y]
                Viz2 = ImgMag[x + 1, y]
            elif (TetaPI > 112.5 and TetaPI < 157.5) or (TetaPI > 292.5 and TetaPI < 337.5):  # Angulos próximos a 135 ou próximos a 315 - diagonal superior esquerda e inferior direita
                Viz1 = ImgMag[x - 1, y - 1]
                Viz2 = ImgMag[x + 1, x + 1]
            # 2º condiçao: Limiar adaptativo # A magnitude de x,y deve ser maior que um Limiar M | LimiarM = sum(ImgMag(x,y) * abs(Viz1 - Viz2)) / sum(abs(Viz1 - Viz2))
            dif = np.abs(Viz1 - Viz2)
            if (dif != 0):
                sum1 = sum1 + (MagPI * dif)
                sum2 = sum2 + dif
    return (sum1 / sum2)

# Recortar a imagem em 9 partes e descobrir qual o quadrante de maior magnitude média
def MagnitudeRegiao(Img, ImgRGB, stats, ImgCC):
    print("Cálculo da Região de Interesse através das médias da Magnitude do Gradiente por Região Possível (5 Horizontais e 3 Verticais) (86)")
    altura = Img.shape[0]; largura = Img.shape[1]
    alturaA = int(altura/5); larguraL = int(largura/3)

    ImgQuadrantes = []
    #Fatias na Horizontal
    ImgQuadrantes.append(Img[0:alturaA, 0:largura])                     #0
    ImgQuadrantes.append(Img[alturaA + 1 : alturaA * 2, 0:largura])     #1
    ImgQuadrantes.append(Img[alturaA * 2 + 1 : alturaA * 3, 0:largura]) #2
    ImgQuadrantes.append(Img[alturaA * 3 + 1: alturaA * 4, 0:largura])  #3
    ImgQuadrantes.append(Img[alturaA * 4 + 1: altura, 0:largura])       #4

    #Fatias na Vertical
    ImgQuadrantes.append(Img[0:altura, 0 : larguraL])                   #5
    ImgQuadrantes.append(Img[0:altura, larguraL + 1 : larguraL * 2])    #6
    ImgQuadrantes.append(Img[0:altura, larguraL * 2 + 1 : largura])     #7

    count1 = 0; count2 = 0; count3 = 0; count3 = 0; count4 = 0; count5 = 0; count6 = 0; count7 = 0; count8 = 0;
    MediaMag1 = 0; MediaMag2 = 0; MediaMag3 = 0; MediaMag4 = 0; MediaMag5 = 0; MediaMag6 = 0; MediaMag7 = 0; MediaMag8 = 0;
    SumDensity1 = 0; SumDensity2 = 0; SumDensity2 = 0; SumDensity3 = 0; SumDensity4 = 0; SumDensity5 = 0; SumDensity6 = 0; SumDensity7 = 0; SumDensity8 = 0;

    for CC in stats[1:]:
        XMin = CC[cv2.CC_STAT_TOP]; XMax = CC[cv2.CC_STAT_TOP] + CC[cv2.CC_STAT_HEIGHT]  # I inicial e final do BB
        YMin = CC[cv2.CC_STAT_LEFT]; YMax = CC[cv2.CC_STAT_LEFT] + CC[cv2.CC_STAT_WIDTH]  # J inicial e final do BB
        if XMin < alturaA:
            count1 += 1
            MediaMag1 = MediaMag1 + np.mean(Img[XMin:XMax, YMin:YMax])
            SumDensity1 = SumDensity1 + CC[cv2.CC_STAT_AREA]
        elif XMin > alturaA + 1 and XMin < alturaA * 2:
            count2 += 1
            MediaMag2 = MediaMag2 + np.mean(Img[XMin:XMax, YMin:YMax])
            SumDensity2 = SumDensity2 + CC[cv2.CC_STAT_AREA]
        elif XMin > alturaA * 2 + 1 and XMin < alturaA * 3:
            count3 += 1
            MediaMag3 = MediaMag3 + np.mean(Img[XMin:XMax, YMin:YMax])
            SumDensity3 = SumDensity3 + CC[cv2.CC_STAT_AREA]
        elif XMin > alturaA * 3 + 1 and XMin < alturaA * 4:
            count4 += 1
            MediaMag4 = MediaMag4 + np.mean(Img[XMin:XMax, YMin:YMax])
            SumDensity4 = SumDensity4 + CC[cv2.CC_STAT_AREA]
        elif XMin > alturaA * 4 + 1:
            count5 += 1
            MediaMag5 = MediaMag5 + np.mean(Img[XMin:XMax, YMin:YMax])
            SumDensity5 = SumDensity5 + CC[cv2.CC_STAT_AREA]
        if YMin < larguraL:
            count6 += 1
            MediaMag6 = MediaMag6 + np.mean(Img[XMin:XMax, YMin:YMax])
            SumDensity6 = SumDensity6 + CC[cv2.CC_STAT_AREA]
        elif YMin > larguraL + 1 and YMin < larguraL * 2:
            count7 += 1
            MediaMag7 = MediaMag1 + np.mean(Img[XMin:XMax, YMin:YMax])
            SumDensity7 = SumDensity7 + CC[cv2.CC_STAT_AREA]
        elif YMin > larguraL * 2 + 1:
            count8 += 1
            MediaMag8 = MediaMag8 + np.mean(Img[XMin:XMax, YMin:YMax])
            SumDensity8 = SumDensity8 + CC[cv2.CC_STAT_AREA]
    print("Counts: ", count1, count2, count3, count4, count5, count6, count7, count8)
    print("MediasMag", MediaMag1, MediaMag2, MediaMag3, MediaMag4, MediaMag5, MediaMag6, MediaMag7, MediaMag8)
    print("SumDensity", SumDensity1, SumDensity2, SumDensity3, SumDensity4, SumDensity5, SumDensity6, SumDensity7, SumDensity8)
    Temp = [count1, count2, count3, count4, count5, count6, count7, count8]
    MaxCount = Temp.index(max(Temp))
    Temp = [MediaMag1, MediaMag2, MediaMag3, MediaMag4, MediaMag5, MediaMag6, MediaMag7, MediaMag8]
    MaxMedia = Temp.index(max(Temp))
    Temp = [SumDensity1, SumDensity2, SumDensity3, SumDensity4, SumDensity5, SumDensity6, SumDensity7, SumDensity8]
    MaxDensity = Temp.index(max(Temp))

    #Cria Mascara para recorte: Maior Count
    AlturaInicial, AlturaFinal, LarguraInicial, LarguraFinal = MascaraROI(MaxCount, altura, largura)
    ImgRGBCopy1 = ImgRGB.copy()
    ImgMascara1 = np.zeros([altura, largura])
    for x in range(AlturaInicial, AlturaFinal):
        for y in range(LarguraInicial, LarguraFinal):
            ImgMascara1[x, y] = 1

    P1 = (LarguraInicial, AlturaInicial)
    P2 = (LarguraFinal, AlturaFinal)
    # desenha retângulos em torno do componente
    cv2.rectangle(ImgRGBCopy1, P1, P2, (0, 0, 255), thickness=2)  # Apenas para Visualização
    #MostraImg("ROI em Vermelho Para MaxCount", ImgRGBCopy1)

    # Cria Mascara para recorte: Maior Media Mag
    AlturaInicial, AlturaFinal, LarguraInicial, LarguraFinal = MascaraROI(MaxMedia, altura, largura)
    ImgRGBCopy2 = ImgRGB.copy()
    ImgMascara2 = np.zeros([altura, largura])
    for x in range(AlturaInicial, AlturaFinal):
        for y in range(LarguraInicial, LarguraFinal):
            ImgMascara2[x, y] = 1

    P1 = (LarguraInicial, AlturaInicial)
    P2 = (LarguraFinal, AlturaFinal)
    # desenha retângulos em torno do componente
    cv2.rectangle(ImgRGBCopy2, P1, P2, (0, 0, 255), thickness=2)  # Apenas para Visualização
    #MostraImg("ROI em Vermelho para MaxMedia", ImgRGBCopy2)

    # Cria Mascara para recorte: Maior Densidade
    AlturaInicial, AlturaFinal, LarguraInicial, LarguraFinal = MascaraROI(MaxDensity, altura, largura)
    ImgRGBCopy3 = ImgRGB.copy()
    ImgMascara3 = np.zeros([altura, largura])
    for x in range(AlturaInicial, AlturaFinal):
        for y in range(LarguraInicial, LarguraFinal):
            ImgMascara3[x, y] = 1

    P1 = (LarguraInicial, AlturaInicial)
    P2 = (LarguraFinal, AlturaFinal)
    # desenha retângulos em torno do componente
    cv2.rectangle(ImgRGBCopy3, P1, P2, (0, 0, 255), thickness=2)  # Apenas para Visualização
    #MostraImg("ROI em Vermelho para MaxDensity", ImgRGBCopy3)

    cv2.imwrite(str(path + imagem + " - 5BBROIMag" + extensao), ImgRGBCopy2)
    #cv2.imwrite(str(path + imagem + "- QuintoBBROIDensidade" + extensao), ImgRGBCopy3)
    return(MaxCount, MaxMedia, MaxDensity)

def MascaraROI(Max, altura, largura):
    alturaA = int(altura/5); larguraL = int(largura/3)
    if Max == 0 or Max == 1 or Max == 2 or Max == 3 or Max == 4:  # Horizontal
        LarguraInicial = 0; LarguraFinal = largura
        if Max == 0:
            AlturaInicial = 0; AlturaFinal = alturaA
        elif Max == 1:
            AlturaInicial = alturaA + 1; AlturaFinal = alturaA * 2
        elif Max == 2:
            AlturaInicial = alturaA * 2 + 1; AlturaFinal = alturaA * 3
        elif Max == 3:
            AlturaInicial = alturaA * 3 + 1; AlturaFinal = alturaA * 4
        else:
            AlturaInicial = alturaA * 4 + 1; AlturaFinal = altura
    else:
        AlturaInicial = 0; AlturaFinal = altura
        if Max == 5:
            LarguraInicial = 0; LarguraFinal = larguraL
        elif Max == 6:
            LarguraInicial = larguraL + 1; LarguraFinal = larguraL * 2
        else:
            LarguraInicial = larguraL * 2 + 1; LarguraFinal = largura

    return AlturaInicial, AlturaFinal, LarguraInicial, LarguraFinal

def RemoveInsideCCs(stats, ImgRGB):
    print("Remove Componentes Internos (356)")
    NovoStats = []; NovoStats.append(stats[0])
    for i, CC in enumerate(stats[1:], start=1):
        A = CC[cv2.CC_STAT_TOP]; B = CC[cv2.CC_STAT_TOP] + CC[cv2.CC_STAT_HEIGHT]  # I inicial e final do BB
        C = CC[cv2.CC_STAT_LEFT]; D = CC[cv2.CC_STAT_LEFT] + CC[cv2.CC_STAT_WIDTH]  # J inicial e final do BB
        Flag = True
        for CCj in stats[1:]:
            XMin = CCj[cv2.CC_STAT_TOP]; XMax = CCj[cv2.CC_STAT_TOP] + CCj[cv2.CC_STAT_HEIGHT]  # I inicial e final do BB
            YMin = CCj[cv2.CC_STAT_LEFT]; YMax = CCj[cv2.CC_STAT_LEFT] + CCj[cv2.CC_STAT_WIDTH]  # J inicial e final do BB
            if A > XMin and B < XMax:
                if C > YMin and D < YMax:
                    #O componente i é um componente filho e não precisa ser rastreado.
                    Flag = False
        if Flag == True:
            NovoStats.append(CC)
    NovoStats = np.asarray(NovoStats)
    ImgRGBCopy = CriaBoundingBoxes(NovoStats, ImgRGB)
    return NovoStats, ImgRGBCopy

def MSER(Img):
    vis = Img.copy()
    mser = cv2.MSER_create()
    regions = mser.detectRegions(Img)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
    cv2.polylines(vis, hulls, 1, (0, 255, 0))
    mask = np.zeros((Img.shape[0], Img.shape[1], 1), dtype=np.uint8)
    for contour in hulls:
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

    # this is used to find only text regions, remaining are ignored
    text_only = cv2.bitwise_and(Img, Img, mask=mask)
    return vis, text_only

def SWT(Img, name):
    #img_in = PIL.Image.open(name+extensao)
    img_out = pillowfight.swt(Img, output_type=pillowfight.SWT_OUTPUT_GRAYSCALE_TEXT)
    img_out.save(path+name+" - 7SWT"+extensao)

    # SWT_OUTPUT_BW_TEXT = 0  # default
    # SWT_OUTPUT_GRAYSCALE_TEXT = 1
    # SWT_OUTPUT_ORIGINAL_BOXES = 2
    #img_out = pillowfight.swt(img_in, output_type=pillowfight.SWT_OUTPUT_ORIGINAL_BOXES)


##### =-=-=-=-=-=-=-=-=-==-=-=- MAIN =-=-=-=-=-=-===-=-=-=- ####
#imgs = ["ImgSimples", "ImgSimples2", "ImgSimples3", "SacredVisitations", "ImgCena1", "LiuImg", "WebBorn"]
#imgs = ["ICDAR-img_1", "ICDAR-img_2", "ICDAR-Img_3", "ICDAR-img_9", "ICDAR-img_12",
#        "ICDAR-img_13", "ICDAR-img_46", "ICDAR-img_59", "ICDAR-img_61",
#        "ICDAR-img_65", "ICDAR-img_73", "ICDAR-img_71", "ICDAR-img_75", "ICDAR-img_76",
#        "ICDAR-img_80", "ICDAR-img_91", "ICDAR-img_96", "ICDAR-img_105", "ICDAR-img_106",
#        "ICDAR-img_107", "ICDAR-img_108", "ICDAR-img_121", "ICDAR-img_125", "ICDAR-img_131",
#        "ICDAR-img_133", "ICDAR-img_134", "ICDAR-img_135"]
imgs = ["12YearsAsSlave", "AliceInWonderland", "AntMan", "Avatar", "Avengers", "AvengersII", "Batman", "BeyondBorders",
        "BlackPanther", "BourneLegacy", "CapitainAmerica", "CitizenKane", "DespicableMe2", "Django", "Eclipse", "EdAstra",
        "Frankenweenie", "Frankie", "GeminiMan", "GoodFellas", "HorribleBosses", "Hustlers", "IceAge4", "IronMan2", "It",
        "Jexi", "Joker", "KingKong", "KillingThemSoftly", "LastStand", "LesMiserables", "Maleficent", "MeetJoeBlack", "MockingjayI",
        "Mortdecai", "Spotlight", "StarWars", "StarWarsIV", "Super8", "Synonyms", "Tangled", "The GoodFather", "TheAviator",
        "TheCurrentWar", "TheHangoover2", "Tintin", "Titanic", "WakingWaves", "WallE", "WonderWoman", "WorldWarZ", "XMen"]
#imgs = ["text_img0002", "text_img0005", "text_img0008", "text_img0011", "text_img0014", "text_img0017", "text_img0020",
#        "text_img0021", "text_img0025", "text_img0028", "text_img0031"]
extensao = ".jpg"
path = "Resultados\\"
f = open(path+"Cartazrs.txt", "w")
f.write("Execução para Cartazes de Filmes: \n\n")
for imagem in imgs:
    ImgRGB, ImgPB = LeImg(imagem + extensao)
    print(imagem)
    f.write("Imagem: " + imagem)
    #Filtro Mediana
    ImgPBM = Mediana(ImgPB)
    #MostraImg("Imagem Cinza Mediana", ImgPBM)

    #Edge Detection Algorithm: Usa Canny completo para detecção de bordas e Filtros de Sobel para Magnitude e Teta (pq canny usa Sobel internamente, embora Prewitt talvez seja melhor pra H e V)
    SobelX, SobelY = Sobel(ImgPBM)
    SobelMag = SobelX + SobelY
    TetaSobel = Teta(SobelX, SobelY)
    ImgCanny = Canny(ImgPBM, SobelMag)
    #MostraImg("Canny", ImgCanny)
    cv2.imwrite(str(path+imagem+" - 0SobelMag"+extensao), SobelMag)
    cv2.imwrite(str(path+imagem+" - 0BordasCanny"+extensao), ImgCanny)

    #Encontra os componetes conectados e os rotula ImgCC
    #Aplica o Bounding Box de acordo com os componentes conectados e retorna a imagem "limpa", apenas com os boxes ImgBB
    ImgCCSimple = ConnectedComponents(ImgCanny)
    ImgCC, stats, centroids, nlabels = ConnectedComponentsStats(ImgCanny)
    ImgRGBCopy = CriaBoundingBoxes(stats, ImgRGB)
    cv2.imwrite(str(path+imagem+" - 1BBSimples"+extensao), ImgRGBCopy)
    f.write("\nTotal de Componentes Conectados: " + str(nlabels))
    #Aplica critérios de seleção e exclusão de BBs de acordo com algumas características da imagem

    # Análise sobre área dos BBs: Envia as estatísticas dos CCs, que tem informação de altura, largura e etc e devolve uma nova lista de stats
    stats, ImgRGBCopy = BBAreasSelection(stats, ImgRGB)
    cv2.imwrite(str(path + imagem + " - 2BBAreas" + extensao), ImgRGBCopy)
    f.write("\nTotal de Componentes Após Remoção de Regiões Não Texto - Áreas: " + str(stats.shape[0]))
    # Remove BB inside
    stats, ImgRGBCopy = RemoveInsideCCs(stats, ImgRGB)
    cv2.imwrite(str(path + imagem + " - 3BBChilds" + extensao), ImgRGBCopy)
    f.write("\nTotal de Componentes Após Redução de BB - InsideBoxes: " + str(stats.shape[0]))
    #Análise de Densidade x Gradiente
    stats, ImgRGBCopy = BBGradientSelection(stats, ImgRGB, SobelMag, TetaSobel, ImgCC)
    cv2.imwrite(str(path+imagem+" - 4BBMags"+extensao), ImgRGBCopy)
    f.write("\nTotal de Componentes Após Remoção de Regiões Não Texto - Gradiente: " + str(stats.shape[0]))
    # Identifica o ROI de acordo com a maior média da magnitude
    MaxCount, MaxMedia, MaxDensity = MagnitudeRegiao(SobelMag, ImgRGBCopy, stats, ImgCC)
    f.write("\nRegião com maior contagem de BBs: " + str(MaxCount) + "; Maior Media Gradiente: " + str(MaxMedia) + "; Maior Densidade: " + str(MaxDensity))
    #NovoStats, pesos = cv2.groupRectangles(list(stats[1:,:4]), 0, 0.)
    #if len(NovoStats) != 0 :
    #    cv2.imwrite(str(path + imagem + " - 6LinkingBBs" + extensao), CriaBoundingBoxes(NovoStats, ImgRGB))

    #vis, text_only = MSER(ImgPBM)
    #cv2.imwrite(str(path + imagem + " - 8MSERImg" + extensao), vis)
    #cv2.imwrite(str(path + imagem + " - 8MSERTextOnly" + extensao), text_only)
    #SWT(ImgCanny, imagem)
    f.write("\n=-=-=-=-=-=- ###### =-=-=-=-=-=-=- \n\n")
f.close()
