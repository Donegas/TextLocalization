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

#Para que o Tesseract Funcione
pyt.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

# Le Imagem RGB e PB. Recebe Caminho e retorna 2 matrizes (1 para RGB ImgRGB e outra Cinza ImgPB)
def LeImg(Img):
    ImgRGB = cv2.imread(Img)
    #ImgPB = cv2.imread(Img, 0)
    ImgPB = cv2.cvtColor(ImgRGB, cv2.COLOR_BGR2GRAY)
    return ImgRGB, ImgPB

###### FUNÇÃO PYTESSERACT DO GOOGLE PARA RECONHECIMENTO AUTOMÁTICO DE CARACTERES
#Recebe Imagem lida (RGB ou PB) e retorna Texto da Imagem
def Tesseract(Img):
    # salva a imagem em um arquivo temporário do Windows para aplicar OCR
    filenameImagem = "{}.jpg".format(os.getpid())
    cv2.imwrite(filenameImagem, Img)
    # carrega a imagem usando a biblioteca PIL/Pillow e aplica OCR
    texto = pyt.image_to_string(Image.open(filenameImagem))
    # deleta arquivo temporário
    os.remove(filenameImagem)
    #print("Texto: " + texto)
    return texto

#Qantização. Recebe a Imagem (RGB ou PB) e a quantidade de clusteres a serem usados.
#Retorna a Imagem Quantizada e os valores de cada cluster escolhido
def Quantizacao(Img, clusters):
    (h, w) = Img.shape[:2]
    # convert the image from the RGB color space to the L*a*b* color space -- since we will be clustering using k-means
    # which is based on the euclidean distance, we'll use the L*a*b* color space where the euclidean distance implies perceptual meaning
    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2LAB)
    # reshape the image into a feature vector so that k-means can be applied
    Img = Img.reshape((Img.shape[0] * Img.shape[1], 3))
    # apply k-means using the specified number of clusters and then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters=clusters)
    labels = clt.fit_predict(Img)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    Img = Img.reshape((h, w, 3))
    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    Img = cv2.cvtColor(Img, cv2.COLOR_LAB2BGR)
    # display the images and wait for a keypress
    #cv2.imshow("image", np.hstack([Img, quant]))
    #cv2.waitKey(0)
    return quant, clt

#Calcula Histograma da Imagem Colorida
def RGBHistograma(Img):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([Img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()

#Histograma da Imagem Cinza
def PBHistograma(Img):
    histr = plt.hist(Img.ravel(), 256, [0, 256]);
    plt.show()

#Não funciona hauhuahauha
def LiuEdgeDetection(Img):
    #DV = (dx dy) Jc.t
    #Matriz Jacobiana [[drdx, dr/dy], [dg/dx, dg/dy], [db/dx, db, dy]] --> Derivadas de primeira ordem dos 3 canais de cor
    #magnitude de DV --> distância euclidiana: DV² = (dx dy) Mc (dx dy).t
    #onde, Mc = Jc.t * Jc = [[Mxx, Mxy],[Mxy, Myy]]
        #Mxx = rx² + gx² + bx²
        #Mxy = rxry + gxgy + bxby
        #Myy = ry² + gy² + by²
    #DV² é a taxa de mudança da imagem da direção dx, dy
    #AutoValor de Mc --> V = ( sqrt( (Mxx + Myy)² - (4 * (Mxx * Myy - Myy²))) + Mxx + Myy) / 2
    #AutoVetor de Mc --> {Mxy - V - Mxx}
    #Direção do Gradiente: Teta = arctan((V-Mxx)/Mxy)
    #Gradiente Magnitude: V(i,j)
    #Gadiente Direction: Teta(i, j)
    altura = Img.shape[0]
    largura = Img.shape[1]
    GradienteMag, GradienteDir = np.zeros([altura, largura]), np.zeros([altura, largura])
    M, Mc = [], []
    #1º Passo: Calcular os Ms: Mxx, Myy, Mxy
    for x in range(0, altura):
        for y in range(0, largura):
            Mxx = Img[x, :, 0]**2 + Img[x, :, 1]**2 + Img[x, :, 2]**2
            Mxy = Img[x, :, 0]*Img[:, y, 0] + Img[x, :, 1]*Img[:, y, 1] + Img[x, :, 2]*Img[:, y, 2]
            Myy = Img[:, y, 0]**2 + Img[:, y, 1]**2 + Img[:, y, 2]**2
            M.append([x, y, Mxx, Mxy, Myy])
            Mc.append([[Mxx, Mxy],[Mxy, Myy]])
            GradienteMag[x][y] = (np.sqrt((Mxx + Myy) ** 2 - (4 * (Mxx * Myy - Myy ** 2))) + Mxx + Myy) / 2
            GradienteDir[x][y] = np.arctan((-Mxx)/Mxy)
    print ("ok")
    cv2.imshow("Mag", GradienteMag)
    cv2.imshow("Dir", GradienteDir)
    cv2.waitKey(0)

#Bordas com Derivadas de Laplace
def Laplace(Img):
    lap = cv2.Laplacian(Img, cv2.CV_64F, ksize=1)
    lap = np.uint8(abs(lap))
    return lap

def Canny(Img, sigma=0.33):
    #1- Redução de Ruído #2 - Calculo do Gradiente; 3 - Supressão de Não Máximos; 4 - Limiar Duplo; 5 - Detecção de Bordas por Hysteresis
    #Calculo de limiares
    v = np.median(Img)
    limiar1 = int(max(0, (1.0 - sigma) * v))
    limiar2 = int(min(255, (1.0 + sigma) * v))
    print("Limiares do Canny: ", limiar1, limiar2)
    #Aplica a detecção de bordas canny do openv
    canny = cv2.Canny(Img, limiar1, limiar2)
    return canny

#Bordas com Derivadas de Sobel
def Sobel(Img):
    sobelx = cv2.Sobel(Img, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(Img, cv2.CV_64F, 0, 1)
    sobelx = np.uint8(abs(sobelx))
    sobely = np.uint8(abs(sobely))
    return sobelx, sobely

def Prewitt(Img):
    # Delimita áreas de variação abruptas de intensidade
    # Aplica Filtros derivativos de Prewitt
    PrewittH = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    PrewittV = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    ImgPrewittH = cv2.filter2D(Img, -1, PrewittH)
    ImgPrewittV = cv2.filter2D(Img, -1, PrewittV)
    return ImgPrewittH, ImgPrewittV

#Baseado em Liu et al e Dissertação Unicamp
def LiuLocalization(ImgX, ImgY):
    altura = ImgX.shape[0]
    largura = ImgX.shape[1]
    Gmag, Gteta = np.zeros([altura, largura, 3]), np.zeros([altura, largura, 3])
    # Gradiente da intensidade da imagem por meio dos filtros derivativos Hh e Hv de prewitt
    #Gradiente da Imagem Original = [ImgPrewittH / ImgPrewittV]
    if len(ImgX.shape) > 2:
        profundidade = ImgX.shape[2]
        for cor in range (0, profundidade):
            for x in range (0, altura):
                for y in range (0, largura):
                    Gmag[x, y, cor] = np.sqrt((ImgX[x, y, cor]**2) + (ImgY[x, y, cor]**2))
                    if ImgX[x, y, cor] == 0: ImgX[x, y, cor] = 1
                    Gteta[x, y, cor] = np.arctan(ImgY[x, y, cor]/ImgX[x, y, cor])
    else: #imagens PB
        for x in range(0, altura):
            for y in range(0, largura):
                Gmag[x, y] = np.sqrt((ImgX[x, y] ** 2) + (ImgY[x, y] ** 2))
                if ImgX[x, y] == 0: ImgX[x, y] = 1
                Gteta[x, y] = np.arctan(ImgY[x, y] / ImgX[x, y])

    return Gmag, Gteta

def LiuLocalization2(ImgPrewittH, ImgPrewittV, Gmag, Gteta):
    #Um pixel de borda é aceito como pixel de borda de um texto quando atende as seguintes condições:
    #A magnitude do gradiente do pixel (i,j) deve ser maior do que a magnitude de ambos os pixels da Vizinhança de 8
    #Vizinhança direta? (i', j') e vizinhança diagonal (i'', j'')
    #relacionados à direção do gradiente do pixel. Assim: Gmg[x][y] > Gmag[x'][y'] e Gmg[x][y] > Gmag[x''][y'']
    #Os dois pixels da vizinhança de 8, cuja magnitue do grandiente é coparada ao pixel i,j são definidos de acordo com o ângulo o gradiente do pixel sob avaliação.
    #Olho para o angulo e comparo com os vizinhos do angulo (sempre em par, x'y' e x''y''
    #se o angulo é 90, os vizinhos estarão diretamente acima e abaixo. São 4 casos possíveis
    #Se a Magnitude do PI for maior que a magnitude dos dois vizinhos, ele é um candidato a bordas textuais
    #Isso pq se espera que as bordas de texto tenham maior magnitude que as demais bordas
    pass

def Gaussiana(Img):
    Img = cv2.GaussianBlur(Img, (3, 3), 0)
    return Img

def Mediana(Img):
    Img = cv2.medianBlur(Img, 9)
    return Img

#Plota Imagens na janela do matplotlib
def PlotaImg(imagens, titulos):
    for i in range(0, len(imagens)-1):
        plt.subplot(5, 6, i+1), plt.imshow(imagens[i][1], "gray")
        plt.title(titulos[i][0])
        #plt.plot()
        plt.xticks([]), plt.yticks([])
    plt.show()

def PlotaImgS(imagens, titulo):
    for i in range(0, len(imagens)-1):
        imagem = cv2.resize(imagens[i][1], None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("Resultados\\" + imgs[op] + imagens[i][0] + ".jpg", imagem)
        cv2.imshow(imagens[i][0], imagem)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

######  MAIN  #####
Imagens = []
imgs = [
    "12YearsAsSlave", "BlackPanther", "CitizenKane", "EdAstra", "Frankie", "GeminiMan", "Hustlers", "It", "Jexi",
    "Joker", "Judy", "MadMax", "Spotlight", "StarWars", "Synonyms", "The GoodFather", "TheCurrentWar", "WakingWaves", "WonderWoman"]
op = 15
extensao = ".jpg"
ImgRGB, ImgPB = LeImg(imgs[op]+extensao)
Imagens.append(["RGB Original", ImgRGB])
Imagens.append(["PB Original", ImgPB])
#Suavização com Gaussiana: Literatura usa Gauss
ImgRGB, ImgPB = Gaussiana(ImgRGB), Gaussiana(ImgPB)
Imagens.append(["RGB Gauss", ImgRGB])
Imagens.append(["PB Gauss", ImgPB])
#Suavização com Mediana: Melhor para ruídos salt and pepper
#ImgRGB, ImgPB = Mediana(ImgRGB), Mediana(ImgPB)

#Conta para "quantidade ideal de cores" ?
#qtdcores = ImgRGB.shape[0]*ImgRGB.shape[1] / ImgRGB.max()+1

#Quantizacao
ImgQuant, Clusteres = Quantizacao(ImgRGB, 16)
Imagens.append(["Quantizada", ImgQuant])

#Histogramas
#RGBHistograma(ImgRGB)
#RGBHistograma(ImgQuant)
#PBHistograma(ImgPB)

#Bordas: Laplace
LapImgRGB = Laplace(ImgRGB)
LapImgQuant = Laplace(ImgQuant)
LapImgPB = Laplace(ImgPB)
Imagens.append(["LaplaceRGB", LapImgRGB])
Imagens.append(["LapaceQuantizado", LapImgQuant])
Imagens.append(["LaplacePB", LapImgPB])

#PlotaImg(Imagens, Titulos)

#Bordas: Sobel
SobelXImgRGB, SobelYImgRGB = Sobel(ImgRGB)
SobelXYImgRGB = cv2.bitwise_or(SobelXImgRGB, SobelYImgRGB)
Imagens.append(["SobelXRGB", SobelXImgRGB])
Imagens.append(["SobelYRGB", SobelYImgRGB])
Imagens.append(["SobelXYRGB", SobelXYImgRGB])

SobelXImgQuant, SobelYImgQuant = Sobel(ImgQuant)
SobelXYImgQuant = cv2.bitwise_or(SobelXImgQuant, SobelYImgQuant)
Imagens.append(["SobelXQuant", SobelXImgQuant])
Imagens.append(["SobelYQuant", SobelYImgQuant])
Imagens.append(["SobelXYQuant", SobelXYImgQuant])

SobelXImgPB, SobelYImgPB  = Sobel(ImgPB)
SobelXYImgPB = cv2.bitwise_or(SobelXImgPB, SobelYImgPB)
Imagens.append(["SobelXImgPB", SobelXImgPB])
Imagens.append(["SobelYImgPB", SobelYImgPB])
Imagens.append(["SobelXYImgPB", SobelXYImgPB])

#Bordas: Canny
CannyImgRGB = Canny(ImgRGB)
CannyImgQuant = Canny(ImgQuant)
CannyImgPB = Canny(ImgPB)
Imagens.append(["CannyImgRGB", CannyImgRGB])
Imagens.append(["CannyImgQuant", CannyImgQuant])
Imagens.append(["CannyImgPB", CannyImgPB])

#Bordas: Prewitt
PrewittXImgRGB, PrewittYImgRGB = Prewitt(ImgRGB)
PrewittXYImgRGB = cv2.bitwise_or(PrewittXImgRGB, PrewittYImgRGB)
Imagens.append(["PrewittXImgRGB", PrewittXImgRGB])
Imagens.append(["PrewittYImgRGB", PrewittYImgRGB])
Imagens.append(["PrewittXYImgRGB", PrewittXYImgRGB])

PrewittXImgQuant, PrewittYImgQuant = Prewitt(ImgQuant)
PrewittXYImgQuant = cv2.bitwise_or(PrewittXImgQuant, PrewittYImgQuant)
Imagens.append(["PrewittXImgQuant", PrewittXImgQuant])
Imagens.append(["PrewittYImgQuant", PrewittYImgQuant])
Imagens.append(["PrewittXYImgQuant", PrewittXYImgQuant])

PrewittXImgPB, PrewittYImgPB = Prewitt(ImgPB)
PrewittXYImgPB = cv2.bitwise_or(PrewittXImgPB, PrewittYImgPB)
Imagens.append(["PrewittXImgPB", PrewittXImgPB])
Imagens.append(["PrewittYImgPB", PrewittYImgPB])
Imagens.append(["PrewittXYImgPB", PrewittXYImgPB])

PlotaImgS(Imagens, Imagens)
#Bordas Liu et al (APENAS GRADIENTE: MAGNITUDE E DIREÇÃO)
#GmagRGB, GtetaRGB = LiuLocalization(ImgRGB)
#GmagQuant, GtetaQuant = LiuLocalization(ImgQuant)
#GmagPB, GtetaPB = LiuLocalization(ImgPB)

#LiuLocalization2(GmagRGB, GtetaRGB)

#Tesseract
#f = open("Resultados\\Tesseract.txt","w+")

#for i in range(0, len(Imagens)-1):
    #texto = Tesseract(Imagens[i][1])
    #titulo = Imagens[i][0]
    #print(titulo, ": ", texto)
    #f.write(titulo)
    #f.write(": ")
    #f.write(texto)
    #f.write("\r\n\n")

#f.close()

'''
print("Tesseract RGB: ", Tesseract(ImgRGB));
print("Tesseract Quant: ", Tesseract(ImgQuant))
print("Tesseract PB: ", Tesseract(ImgPB))

print("Tesseract Laplace RGB: ", Tesseract(LapImgRGB))
print("Tesseract Laplace Quant: ", Tesseract(LapImgQuant))
print("Tesseract Laplace PB: ", Tesseract(LapImgPB))

print("Tesseract Sobel RGB: ", Tesseract(SobelXYImgRGB))
print("Tesseract Sobel Quant: ", Tesseract(SobelXYImgQuant))
print("Tesseract Sobel PB: ", Tesseract(SobelXYImgPB))

print("Tesseract Canny RGB: ", Tesseract(CannyImgRGB))
print("Tesseract Canny Quant: ", Tesseract(CannyImgQuant))
print("Tesseract Canny PB: ", Tesseract(CannyImgPB))

print("Tesseract Prewitt RGB: ", Tesseract(PrewittXYImgRGB))
print("Tesseract Prewitt Quant: ", Tesseract(PrewittXYImgQuant))
print("Tesseract Prewitt PB: ", Tesseract(PrewittXYImgPB))'''
