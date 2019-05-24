# -*- coding: utf-8 -*-
""" Aluno: Leonardo Santos Paulucio
    Data: 24/05/19
    Trabalho 2 de Visão Computacional
"""

import random
import numpy as np
import scipy.optimize as optimize


def transferError(H, x, xi):
    """ Calcule transfer error from points"""

    H = H.reshape((3, 3))
    x_p = np.dot(H, x)
    x_p /= x_p[2]

    return np.square(np.linalg.norm(xi - x_p, axis=0))


def symmetricError(H, x, xi):
    """ Calcule symmetric error from points"""

    H = H.reshape((3,3))
    x_p = np.dot(H, x)
    x_p /= x_p[2]

    xi_p = np.dot(np.linalg.inv(H), xi)
    xi_p /= xi_p[2]

    return np.square(np.linalg.norm(xi - x_p, axis=0) + np.linalg.norm(x - xi_p, axis=0))


def normalizePoints(points):
    """ Function that normalize points
    2d points [xi, yi]:
    First column xi, second column yi
    Return normalized points and transformation matrix
    """

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


def checkSubset(src, dst):
    """ Check points collinearity by calculating triangle area"""

    if len(src) != 4 or len(dst) != 4:
        print("Subset length isn't 4")
        return

    EPSILON = 0.000005
    indexes = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]

    negative = 0

    for i in indexes:
        matA = cart2homo(src[i])
        matB = cart2homo(dst[i])
        areaA = np.linalg.det(matA)
        areaB = np.linalg.det(matB)

        # Check collinearity by calculating the area of triangle.
        if np.abs(areaA) <= EPSILON or np.abs(areaB) <= EPSILON:
            # Collinear points
            return False

        # We check whether the minimal set of points for the homography estimation
        # are geometrically consistent. We check if every 3 correspondences sets
        # fulfills the constraint.
        #
        # The usefullness of this constraint is explained in the paper:
        #
        # "Speeding-up homography estimation in mobile devices"
        # Journal of Real-Time Image Processing. 2013. DOI: 10.1007/s11554-012-0314-1
        # Pablo Marquez-Neila, Javier Lopez-Alberca, Jose M. Buenaposada, Luis Baumela
        negative += areaA * areaB < 0

    if negative != 0 and negative != 4:
        return False

    return True


def getInliersNumber(src, dst, H, tolerance, f_error):
    """ Counting the number of inliers on set """

    mask = np.zeros(len(src))

    src = cart2homo(src).T
    dst = cart2homo(dst).T

    erro = f_error(H, src, dst)
    mask[erro < tolerance] = 1
    return mask


def calcHomography(src, dst, normalize=True):
    """ Calcule homography matrix using DLT algorithm"""

    if normalize:
        # Normalizing points
        src_pts, src_T = normalizePoints(src)
        dst_pts, dst_T = normalizePoints(dst)

    A = createMatrixA(src_pts, dst_pts)
    U, S, V = np.linalg.svd(A)
    H = np.reshape(V[-1], (3, 3))

    if normalize:
        # Denormalizing Ĥ --> H = (T'^-1) x Ĥ x T
        H = np.dot(H, src_T)
        H = np.dot(np.linalg.inv(dst_T), H)

    return H


def RANSAC(src_pts, dst_pts, min_pts_required=4, tolerance=5.0, threshold=0.6, N=1000,
           f_error=transferError, normalize=True):
    """ Ransac algorithm """

    if len(src_pts) < min_pts_required:
        print("Number of src points don't satisfy minimum required.")
        exit()

    if len(src_pts) != len(dst_pts):
        print("Number of src points and dst points don't match.")
        exit()

    NUM_POINTS = len(src_pts)
    H_best = None
    mask_best = None
    num_inliers = 0

    for i in range(0, N):
        goodSubset = False
        while not(goodSubset):
            samples_index = random.sample(range(NUM_POINTS), min_pts_required)
            src_sample = src_pts[samples_index]
            dst_sample = dst_pts[samples_index]
            goodSubset = checkSubset(src_sample, dst_sample)

        H = calcHomography(src_sample, dst_sample, normalize=True)

        src_sample = cart2homo(src_sample).T
        dst_sample = cart2homo(dst_sample).T

        mask = getInliersNumber(src_pts, dst_pts, H, 5.0, f_error)
        inliers = np.count_nonzero(mask)

        if inliers > num_inliers:
            num_inliers = inliers
            H_best = H
            mask_best = mask

    return H_best, mask_best


def cart2homo(points):
    """ Converting cartesian points to homogeneous coordinates"""
    return np.append(points, np.ones((len(points), 1)), axis=1)


def createMatrixA(x, y):
    """ Creating A matrix for DLT """

    N = len(x)
    A = np.zeros((2*N, 9))

    for i, (p, pi) in enumerate(zip(x, y)):
        i *= 2
        x, y = p
        xp, yp = pi
        # first line
        A[i, 0:3] = np.insert(-p, 2, -1)  # -x -y -1 0 0 0 x*xp y*xp xp
        A[i, -3:] = np.array([x*xp, y*xp, xp])
        # second line
        A[i+1, 3:6] = np.insert(-p, 2, -1)  # 0 0 0 -x -y -1 x*yp y*yp yp
        A[i+1, -3:] = np.array([x*yp, y*yp, yp])

    return A


def findHomography(src, dst, type="RANSAC", reprojectionErrorThreshold=5.0):
    """ Find homography between two points set """

    # Finding best H with Ransac
    H, mask = RANSAC(src, dst)

    idx = (mask == 1)

    # Calculating homography for all inliers
    H = calcHomography(src[idx], dst[idx], normalize=True)

    src = cart2homo(src[idx]).T
    dst = cart2homo(dst[idx]).T

    # Optimizing H with least squares
    solucao = optimize.least_squares(symmetricError, H.reshape(-1), method="lm", args=(src, dst),
                                     verbose=False, max_nfev=50000)
    H = solucao.x.reshape((3, 3))
    print("Number of inliers: {}".format(np.count_nonzero(mask)))
    print("H:\n {}".format(H))

    return H, mask
