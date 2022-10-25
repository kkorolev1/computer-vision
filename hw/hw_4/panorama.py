import numpy as np
from skimage.feature import ORB, match_descriptors
from skimage.color import rgb2gray
from skimage.transform import ProjectiveTransform
from skimage.transform import warp
from skimage.filters import gaussian
from numpy.linalg import inv

DEFAULT_TRANSFORM = ProjectiveTransform


def find_orb(img, n_keypoints=500):
    """Find keypoints and their descriptors in image.

    img ((W, H, 3)  np.ndarray) : 3-channel image
    n_keypoints (int) : number of keypoints to find

    Returns:
        (N, 2)  np.ndarray : keypoints
        (N, 256)  np.ndarray, type=np.bool  : descriptors
    """
    descriptor_extractor = ORB(n_keypoints=n_keypoints)
    descriptor_extractor.detect_and_extract(rgb2gray(img))
    return descriptor_extractor.keypoints, descriptor_extractor.descriptors


def center_and_normalize_points(points):
    """Center the image points, such that the new coordinate system has its
    origin at the centroid of the image points.

    Normalize the image points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(2).

    points ((N, 2) np.ndarray) : the coordinates of the image points

    Returns:
        (3, 3) np.ndarray : the transformation matrix to obtain the new points
        (N, 2) np.ndarray : the transformed image points
    """
    pointsh = np.row_stack([points.T, np.ones((points.shape[0]), )])
    centroid = points.mean(axis=0)

    N = np.sqrt(2) / (np.sum(np.sqrt((points[:, 0] - centroid[0]) ** 2 + (points[:, 1] - centroid[1]) ** 2)) / points.shape[0])
    matrix = np.array([[N, 0, -N*centroid[0]],
                       [0, N, -N*centroid[1]],
                       [0, 0, 1]])
    return matrix, (matrix @ pointsh)[:2, :].T


def find_homography(src_keypoints, dest_keypoints):
    """Estimate homography matrix from two sets of N (4+) corresponding points.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates

    Returns:
        ((3, 3) np.ndarray) : homography matrix
    """
    src_matrix, src = center_and_normalize_points(src_keypoints)
    dest_matrix, dest = center_and_normalize_points(dest_keypoints)

    # src_matrix - M
    # dest_matrix - M'

    convert_to_ax = lambda old, new: np.array([-old[0],-old[1],-1,0,0,0,old[0]*new[0],old[1]*new[0],new[0]])
    convert_to_ay = lambda old, new: np.array([0,0,0,-old[0],-old[1],-1,old[0]*new[1],old[1]*new[1],new[1]])

    A = []
    for s, d in zip(src, dest):
        A.append(convert_to_ax(s, d))
        A.append(convert_to_ay(s, d))
    A = np.array(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1,:]
    H = h.reshape(3, 3)

    return inv(dest_matrix) @ H @ src_matrix


def ransac_transform(src_keypoints, src_descriptors, dest_keypoints, dest_descriptors, max_trials=1000, residual_threshold=5, return_matches=False):
    """Match keypoints of 2 images and find ProjectiveTransform using RANSAC algorithm.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    src_descriptors ((N, 256) np.ndarray) : source descriptors
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates
    dest_descriptors ((N, 256) np.ndarray) : destination descriptors
    max_trials (int) : maximum number of iterations for random sample selection.
    residual_threshold (float) : maximum distance for a data point to be classified as an inlier.
    return_matches (bool) : if True function returns matches

    Returns:
        skimage.transform.ProjectiveTransform : transform of source image to destination image
        (Optional)(N, 2) np.ndarray : inliers' indexes of source and destination images
    """
    matches = match_descriptors(src_descriptors, dest_descriptors)

    src_keypoints = src_keypoints[matches[:,0]]
    dest_keypoints = dest_keypoints[matches[:,1]]

    best_inliers = None
    num_best_inliers = 0

    for iter in range(max_trials):
        indices = np.random.choice(src_keypoints.shape[0], 4)
        H = find_homography(src_keypoints[indices,:], dest_keypoints[indices,:])

        src_h = np.row_stack([src_keypoints.T, np.ones((src_keypoints.shape[0]), )])

        new_h = H @ src_h
        new_h[0] /= new_h[2]
        new_h[1] /= new_h[2]

        new_h = new_h[:2].T

        mask = np.linalg.norm(new_h - dest_keypoints, axis=1) < residual_threshold

        if mask.sum() >= num_best_inliers:
            num_best_inliers = mask.sum()
            best_inliers = mask

    H = find_homography(src_keypoints[best_inliers], dest_keypoints[best_inliers])
    transform = ProjectiveTransform(H)

    if return_matches:
        return transform, matches[best_inliers]

    return transform


def find_simple_center_warps(forward_transforms):
    """Find transformations that transform each image to plane of the central image.

    forward_transforms (Tuple[N]) : - pairwise transformations

    Returns:
        Tuple[N + 1] : transformations to the plane of central image
    """
    image_count = len(forward_transforms) + 1
    center_index = (image_count - 1) // 2

    result = [None] * image_count
    result[center_index] = DEFAULT_TRANSFORM()

    for i in range(center_index + 1, image_count):
        result[i] = result[i-1] + ProjectiveTransform(inv(forward_transforms[i-1].params))

    for i in range(center_index - 1, -1, -1):
        result[i] = result[i+1] + forward_transforms[i]

    return tuple(result)


def get_corners(image_collection, center_warps):
    """Get corners' coordinates after transformation."""
    for img, transform in zip(image_collection, center_warps):
        height, width, _ = img.shape
        corners = np.array([[0, 0],
                            [height, 0],
                            [height, width],
                            [0, width]])

        yield transform(corners)[:, ::-1]


def get_min_max_coords(corners):
    """Get minimum and maximum coordinates of corners."""
    corners = np.concatenate(corners)
    return corners.min(axis=0), corners.max(axis=0)


def get_final_center_warps(image_collection, simple_center_warps):
    """Find final transformations.

        image_collection (Tuple[N]) : list of all images
        simple_center_warps (Tuple[N])  : transformations unadjusted for shift

        Returns:
            Tuple[N] : final transformations
        """
    corners = tuple(get_corners(image_collection, simple_center_warps))
    min_coord, max_coord = get_min_max_coords(corners)
    width = int(max_coord[0] - min_coord[0]) + 1
    height = int(max_coord[1] - min_coord[1]) + 1
    matrix = np.array([[1, 0, -min_coord[1]],
                       [0, 1, -min_coord[0]],
                       [0, 0, 1]])
    return tuple([tr + ProjectiveTransform(matrix) for tr in simple_center_warps]), (height, width)


def rotate_transform_matrix(transform):
    """Rotate matrix so it can be applied to row:col coordinates."""
    matrix = transform.params[(1, 0, 2), :][:, (1, 0, 2)]
    return type(transform)(matrix)


def warp_image(image, transform, output_shape):
    """Apply transformation to an image and its mask

    image ((W, H, 3)  np.ndarray) : image for transformation
    transform (skimage.transform.ProjectiveTransform): transformation to apply
    output_shape (int, int) : shape of the final pano

    Returns:
        (W, H, 3)  np.ndarray : warped image
        (W, H)  np.ndarray : warped mask
    """
    # your code here
    transformed_image = warp(image, rotate_transform_matrix(transform), output_shape=output_shape)
    mask = (transformed_image[...,0] > 0) | (transformed_image[...,1] > 0) | (transformed_image[...,2] > 0)
    return transformed_image, mask.astype(bool)


def merge_pano(image_collection, final_center_warps, output_shape):
    """ Merge the whole panorama

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano

    Returns:
        (output_shape) np.ndarray: final pano
    """
    result = np.zeros(output_shape + (3,))
    result_mask = np.zeros(output_shape, dtype=np.bool8)

    for image, transform in zip(image_collection, final_center_warps):
        transformed_image, mask = warp_image(image, ProjectiveTransform(inv(transform)), output_shape)
        mask &= ~result_mask
        transformed_image = np.dstack((transformed_image[...,0] * mask, transformed_image[...,1] * mask, transformed_image[...,2] * mask))
        result_mask |= mask
        result += transformed_image
    return np.clip(np.rint(result * 255), 0, 255).astype(np.uint8)


def get_gaussian_pyramid(image, n_layers, sigma):
    """Get Gaussian pyramid.

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Gaussian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Gaussian pyramid

    """
    # your code here
    pass


def get_laplacian_pyramid(image, n_layers, sigma):
    """Get Laplacian pyramid

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Laplacian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Laplacian pyramid
    """
    # your code here
    pass


def merge_laplacian_pyramid(laplacian_pyramid):
    """Recreate original image from Laplacian pyramid

    laplacian pyramid: tuple of np.array (h, w, 3)

    Returns:
        np.array (h, w, 3)
    """
    return sum(laplacian_pyramid)


def increase_contrast(image_collection):
    """Increase contrast of the images in collection"""
    result = []

    for img in image_collection:
        img = img.copy()
        for i in range(img.shape[-1]):
            img[:, :, i] -= img[:, :, i].min()
            img[:, :, i] /= img[:, :, i].max()
        result.append(img)

    return result


def gaussian_merge_pano(image_collection, final_center_warps, output_shape, n_layers, image_sigma, merge_sigma):
    """ Merge the whole panorama using Laplacian pyramid

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano
    n_layers (int) : number of layers in Laplacian pyramid
    image_sigma (int) :  sigma for Gaussian filter for images
    merge_sigma (int) : sigma for Gaussian filter for masks

    Returns:
        (output_shape) np.ndarray: final pano
    """
    # your code here
    pass

def cylindrical_inverse_map(coords, h, w, scale):
    """Function that transform coordinates in the output image
    to their corresponding coordinates in the input image
    according to cylindrical transform.

    Use it in skimage.transform.warp as `inverse_map` argument

    coords ((M, 2) np.ndarray) : coordinates of output image (M == col * row)
    h (int) : height (number of rows) of input image
    w (int) : width (number of cols) of input image
    scale (int or float) : scaling parameter

    Returns:
        (M, 2) np.ndarray : corresponding coordinates of input image (M == col * row) according to cylindrical transform
    """
    # your code here
    pass

def warp_cylindrical(img, scale=None, crop=True):
    """Warp image to cylindrical coordinates

    img ((H, W, 3)  np.ndarray) : image for transformation
    scale (int or None) : scaling parameter. If None, defaults to W * 0.5
    crop (bool) : crop image to fit (remove unnecessary zero-padding of image)

    Returns:
        (H, W, 3)  np.ndarray : warped image (H and W may differ from original)
    """
    # your code here
    pass


# Pick a good scale value for the 5 test image sets
cylindrical_scales = {
    0: None,
    1: None,
    2: None,
    3: None,
    4: None,
}
