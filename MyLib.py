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
    # Lembrar q passa por referencia os pontos

    # x - x_bar
    # x_pts -= bar[0]
    # y - y_bar
    # y_pts -= bar[1]

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
    return norm_pts, T


def collinear(x1, y1, x2, y2, x3, y3):
    """ Calculation the area of
        triangle. We have skipped
        multiplication with 0.5 to
        avoid floating point computations """
    a = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)

    if (a == 0):
        print("Yes")
    else:
        print("No")
# eps=0.0000005

# Slope based solution to check if three
# points are collinear.

# function to check if
# point collinear or not
def collinear(x1, y1, x2, y2, x3, y3):

    if ((y3 - y2)*(x2 - x1) == (y2 - y1)*(x3 - x2)):
        print("Yes")
    else:
        print("No")


def getInliersNumber(src, dst, H, tolerance, f_error):

    mask = np.zeros(len(src))

    src = cart2homo(src).T
    dst = cart2homo(dst).T

    erro = f_error(src, dst, H)
    mask[erro < tolerance] = 1
    return mask


def RANSAC(src_pts, dst_pts, min_pts_required=4, tolerance=5.0, threshold=0.6, N=1000, f_error=transferError):

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
        samples_index = random.sample(range(NUM_POINTS), min_pts_required)
        src_sample = src_pts[samples_index]
        dst_sample = dst_pts[samples_index]

        # Normalizing points
        # src_norm, src_T = normalizePoints(src_sample)
        # dst_norm, dst_T = normalizePoints(dst_sample)
        src_norm = src_sample
        dst_norm = dst_sample

        A = createMatrixA(src_norm, dst_norm)

        U, S, V = np.linalg.svd(A)

        H = np.reshape(V[-1], (3, 3))

        src_sample = cart2homo(src_sample).T
        dst_sample = cart2homo(dst_sample).T

        # Denormalizing H --> H = (T'^-1) x Ĥ x T
        # H = np.dot(H, src_T)
        # H = np.dot(np.linalg.inv(dst_T), H)
        mask = getInliersNumber(src_pts, dst_pts, H, 5.0, f_error)
        inliers = np.count_nonzero(mask)

        if inliers > num_inliers:
            num_inliers = inliers
            H_best = H
            mask_best = mask

    print(num_inliers)
    print(H_best)
    exit()


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

    RANSAC(src, dst)

    exit()

    # [[ 3.88128952e-01  5.19597837e-02  1.20670565e+01]
    # [-2.49799069e-01  7.69568940e-01  2.23373460e+02]
    # [-3.28739078e-04  1.58800881e-04  1.00000000e+00]]

    # return M, mask
