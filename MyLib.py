# -*- coding: utf-8 -*-
""" Aluno: Leonardo Santos Paulucio
    Data: 24/05/19
    Trabalho 2 de Visão Computacional
"""

import numpy as np
import random


def transferError(x, xi, H):
    x_p = np.dot(H, x)
    x_p /= x_p[2]

    return np.linalg.norm(xi - x_p, axis=0)


def symmetricError(x, xi, H):
    x_p = np.dot(H, x)
    x_p /= x_p[2]

    xi_p = np.dot(np.linalg.inv(H), xi)
    xi_p /= xi_p[2]

    return np.linalg.norm(xi - x_p, axis=0) + np.linalg.norm(x - xi_p, axis=0)


def normalizePoints(points):
    # 2d points [x,y]: First column xi, second column yi
    # Centroid
    center = np.mean(points, axis=0)  # [x_bar, y_bar]

    norm_pts = points - center  # x - xc  y - yc
    rho_bar = np.sqrt(np.square(norm_pts).sum(axis=1)).mean()  # sqrt((x-xc)^2 + (y-yc)^2) / n
    s = np.sqrt(2) / rho_bar
    norm_pts *= s

    T = np.array([[s, 0, -s*center[0]],
                  [0, s, -s*center[1]],
                  [0, 0,           1]])

    return norm_pts, T


def isCollinear(points):
    """ Check collinearity by calculating the area of triangle. """

    EPSILON = 0.000005
    indexes = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]

    for i in indexes:
        matrix = cart2homo(points[i])
        area = np.linalg.det(matrix)
        if np.abs(area) <= EPSILON:
            return True
    return False


def getInliersNumber(src, dst, H, tolerance, f_error):

    mask = np.zeros(len(src))

    src = cart2homo(src).T
    dst = cart2homo(dst).T

    erro = f_error(src, dst, H)
    mask[erro < tolerance] = 1
    return mask


def calcHomography(src, dst, normalize=True):

    if normalize:
        # Normalizing points
        src_pts, src_T = normalizePoints(src)
        dst_pts, dst_T = normalizePoints(dst)

    A = createMatrixA(src_pts, dst_pts)
    U, S, V = np.linalg.svd(A)
    H = np.reshape(V[-1], (3, 3))

    if normalize:
        # Denormalizing H --> H = (T'^-1) x Ĥ x T
        H = np.dot(H, src_T)
        H = np.dot(np.linalg.inv(dst_T), H)

    return H


def RANSAC(src_pts, dst_pts, min_pts_required=4, tolerance=5.0, threshold=0.6, N=2000,
           f_error=transferError, normalize=True):

    if len(src_pts) < min_pts_required:
        print("Number of src points don't satisfy minimum required.")
        return None

    if len(src_pts) != len(dst_pts):
        print("Number of src points and dst points don't match.")
        return None

    NUM_POINTS = len(src_pts)
    H_best = None
    mask_best = None
    num_inliers = 0

    for i in range(0, N):
        collinear = True
        while collinear:
            samples_index = random.sample(range(NUM_POINTS), min_pts_required)
            src_sample = src_pts[samples_index]
            collinear = isCollinear(src_sample)

        dst_sample = dst_pts[samples_index]

        H = calcHomography(src_sample, dst_sample, normalize=True)

        src_sample = cart2homo(src_sample).T
        dst_sample = cart2homo(dst_sample).T

        mask = getInliersNumber(src_pts, dst_pts, H, 5.0, f_error)
        inliers = np.count_nonzero(mask)

        if inliers > num_inliers:
            num_inliers = inliers
            H_best = H
            mask_best = mask

    print(num_inliers)
    return H_best, mask_best


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
        A[i, 0:3] = np.insert(-p, 2, -1)  # -x -y -1 0 0 0 x*xp y*xp xp
        A[i, -3:] = np.array([x*xp, y*xp, xp])
        # segunda linha
        A[i+1, 3:6] = np.insert(-p, 2, -1)  # 0 0 0 -x -y -1 x*yp y*yp yp
        A[i+1, -3:] = np.array([x*yp, y*yp, yp])

    return A


def findHomography(src, dst, type="RANSAC", reprojectionErrorThreshold=5.0):

    H, mask = RANSAC(src, dst)

    idx = (mask == 1)

    # NORMALIZAR PONTOS
    H = calcHomography(src[idx], dst[idx], normalize=True)

    # OTIMIZAR
    # H^-1 x dst, H para funcao de otimizacao

    #optimize(src, dst, H)

    # [[ 3.88128952e-01  5.19597837e-02  1.20670565e+01]
    # [-2.49799069e-01  7.69568940e-01  2.23373460e+02]
    # [-3.28739078e-04  1.58800881e-04  1.00000000e+00]]

    return H, mask
