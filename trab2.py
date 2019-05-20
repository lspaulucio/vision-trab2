# -*- coding: utf-8 -*-
""" Aluno: Leonardo Santos Paulucio
    Data: 24/05/19
    Trabalho 2 de Visão Computacional

Neste trabalho vocês deverão calcular a homografia entre duas imagens, usando:

- RANSAC para eliminar os outliers

- um processo de minimização/otimização para se obter uma estimativa mais precisa da homografia

- um método de reprojeção das imagens para se obter a Imagem 1 no referencial da Imagem 2 e vice-versa.

Vocês deverão testar o algoritmo desenvolvido com os casos que vimos na aula de Homografia no Google Colab.

O trabalho poderá ser feito em dupla e deverá ser entregue/apresentado até o dia 24/05/2019.

Vocês deverão enviar o trabalho para raquel@ele.ufes.br até a data prevista e agendar a apresentação.
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import imutils


def euclidianDistance(x,y):
    return np.sqrt((x - y)**2).sum()


def normalizePoints(points):
    # Lembrar q passa por referencia os pontos

    # x - x_bar
    # x_pts -= bar[0]
    # y - y_bar
    # y_pts -= bar[1]

    # 2d points [x,y]: First column xi, second column yi
    bar = np.mean(points, axis=0) # [x_bar, y_bar]

    points -= bar
    rho_bar = np.sqrt(np.square(points).sum(axis=1)).mean()
    s = np.sqrt(2) / rho_bar
    points *= s

    T = np.array([[s, 0, -s*bar[0]],
                  [0, s, -s*bar[1]],
                  [0, 0,     1   ]])

    # % compute the translation
    # x_bar = mean(x, 2);
    #
    # % center the points
    # % faster than xc = x - repmat(x_bar, 1, size(x, 2));
    # xc(1, :) = x(1, :) - x_bar(1);
    # xc(2, :) = x(2, :) - x_bar(2);
    # % compute the average point distance
    # rho = sqrt(sum(xc.^2, 1));  sqrt(x - x_bar)
    # rho_bar = mean(rho);        mean(sqrt(x - x_bar))
    #
    # % compute the scale factor
    # s = sqrt(2)/rho_bar;
    #
    # % scale the points
    # xn = s*xc;
    #
    # % compute the transformation matrix
    # T = [s 0 -s*x_bar(1); 0 s -s*x_bar(2); 0 0 1];


def cart2homo(points):
    return np.append(points, np.ones((len(points), 1)), axis=1)


def createMatrixA(x, y):

    N = len(x)
    A = np.zeros((2*N, 9))

    for i, (p, pi) in enumerate(zip(x, y)):
        i *= 2
        x, y = p
        xp, yp = pi
        # primeira linha
        A[i, 0:3] = np.insert(-p, 2, -1)  # -x -y -1
        A[i, -3:] = np.array([x*xp, y*xp, xp])
        # segunda linha
        A[i+1, 3:6] = np.insert(-p, 2, -1)
        A[i+1, -3:] = np.array([x*yp, y*yp, yp])

    return A

def findHomography(src, dst, type="RANSAC", reprojectionErrorThreshold=5.0):

    NUM_ITER = 1000
    NUM_POINTS = 4
    error = 10000000000000000000
    for i in range(0, NUM_ITER):
        # samples_index = np.random.randint(0, len(src), NUM_POINTS)
        # samples_index = np.array([0,1,2,3])
        if i+3 > 288:
            break
        samples_index = np.array([i, i+1, i+2, i+3])
        src_sample = src[samples_index]
        dst_sample = dst[samples_index]

        A = createMatrixA(src_sample, dst_sample)

        [U, S, V] = np.linalg.svd(np.dot(A.T, A))

        H = np.reshape(V[:,-1], (3, 3))

        src_sample = cart2homo(src_sample)
        dst_sample = cart2homo(dst_sample)

        proj_src = np.dot(H, src_sample.T)
        proj_src /= proj_src[2]

        proj_dst = np.dot(np.linalg.inv(H), dst_sample.T)
        proj_dst /= proj_dst[2]

        # symmetric transfer error
        erro = euclidianDistance(dst_sample.T, proj_src) + euclidianDistance(src_sample.T, proj_dst)
        if erro < error:
            error = erro

    print(error)
    exit()

 # [[ 3.88128952e-01  5.19597837e-02  1.20670565e+01]
 # [-2.49799069e-01  7.69568940e-01  2.23373460e+02]
 # [-3.28739078e-04  1.58800881e-04  1.00000000e+00]]




    return M, mask


# Using SIFT to estimate Homography between images and to warp the first image

MIN_MATCH_COUNT = 10
img1 = cv.imread('images/outdoors01.jpg',0)          # queryImage
img2 = cv.imread('images/outdoors02.jpg',0) # trainImage

#img1 = imutils.rotate_bound(img1,180)

# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)


# FLANN stands for Fast Library for Approximate Nearest Neighbors.
# It contains a collection of algorithms optimized for fast nearest neighbor
# search in large datasets and for high dimensional features.
# It works faster than BFMatcher for large datasets.
# The variable index_params specifies the algorithm to be used, its related parameters etc.
# For algorithms like SIFT, SURF etc. you can pass following:
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# The variable search_params specifies the number of times the trees in the index should
# be recursively traversed. Higher values gives better precision, but also takes more time.
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)
#bf = cv.BFMatcher()
#matches = bf.knnMatch(des1,des2,k=2)plt.imshow(img3, 'gray')


# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
    M, mask = findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    exit()
#     matchesMask = mask.ravel().tolist()
#
#     img4 = cv.warpPerspective(img1, M, (img1.shape[1],img1.shape[0])) #, None) #, flags[, borderMode[, borderValue]]]]	)
#
# else:
#     print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
#     matchesMask = None
#
# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = None,
#                    matchesMask = matchesMask, # draw only inliers
#                    flags = 2)
# img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
#
# fig = plt.figure(figsize=(25,10))
# ax1 = fig.add_subplot(2,2,1)
# plt.imshow(img3, 'gray')
# ax1 = fig.add_subplot(2,2,2)
# plt.title('First image')
# plt.imshow(img1,'gray')
# ax1 = fig.add_subplot(2,2,3)
# plt.title('Second image')
# plt.imshow(img2,'gray')
# ax1 = fig.add_subplot(2,2,4)
# plt.title('First image after transformation')
# plt.imshow(img4,'gray')
#
# plt.show()
