#!/usr/bin/env python3

import os
import shutil
import traceback

import panorama
import plots

from skimage import io
import numpy as np


def process(i):
    prefix = f'./results/{i:02d}_'
    np.random.seed(i)

    # ------------------------------------------------------------------------------------------
    # Part 0
    # ------------------------------------------------------------------------------------------

    pano_image_collection = io.ImageCollection(f'imgs/{i:02d}/*',
                                               load_func=lambda f: io.imread(f).astype(np.float64) / 255)


    plots.plot_collage(prefix + '01_collage.jpeg', pano_image_collection, title=f"Image collection size: {len(pano_image_collection)}")

    # ------------------------------------------------------------------------------------------
    # Part 1
    # ------------------------------------------------------------------------------------------

    img = pano_image_collection[0]
    keypoints, descriptors = panorama.find_orb(img)

    plots.plot_keypoints(prefix + '02_keypoints.jpeg', img, keypoints)

    # ------------------------------------------------------------------------------------------
    # Part 2 and 3
    # ------------------------------------------------------------------------------------------

    src, dest = pano_image_collection[0], pano_image_collection[1]
    src_keypoints, src_descriptors = panorama.find_orb(src)
    dest_keypoints, dest_descriptors = panorama.find_orb(dest)

    robust_transform, matches = panorama.ransac_transform(src_keypoints, src_descriptors, dest_keypoints, dest_descriptors, return_matches=True)

    plots.plot_inliers(prefix + '03_inliers.jpeg', src, dest, src_keypoints, dest_keypoints, matches)

    # ------------------------------------------------------------------------------------------
    # Part 4
    # ------------------------------------------------------------------------------------------

    keypoints, descriptors = zip(*(panorama.find_orb(img) for img in pano_image_collection))
    forward_transforms = tuple(panorama.ransac_transform(src_kp, src_desc, dest_kp, dest_desc)
                               for src_kp, src_desc, dest_kp, dest_desc
                               in zip(keypoints[:-1], descriptors[:-1], keypoints[1:], descriptors[1:]))


    simple_center_warps = panorama.find_simple_center_warps(forward_transforms)
    corners = tuple(panorama.get_corners(pano_image_collection, simple_center_warps))
    min_coords, max_coords = panorama.get_min_max_coords(corners)
    center_img = pano_image_collection[(len(pano_image_collection) - 1) // 2]

    plots.plot_warps(prefix + '04_simple_warps.jpeg', corners, min_coords=min_coords, max_coords=max_coords, img=center_img)


    final_center_warps, output_shape = panorama.get_final_center_warps(pano_image_collection, simple_center_warps)
    corners = tuple(panorama.get_corners(pano_image_collection, final_center_warps))

    plots.plot_warps(prefix + '05_final_warps.jpeg', corners, output_shape=output_shape)

    # ------------------------------------------------------------------------------------------
    # Part 5
    # ------------------------------------------------------------------------------------------

    result = panorama.merge_pano(pano_image_collection, final_center_warps, output_shape)

    plots.plot_result(prefix + '06_base_pano.jpeg', result)

    return
    # ------------------------------------------------------------------------------------------
    # Part 6
    # ------------------------------------------------------------------------------------------

    img = pano_image_collection[0]

    laplacian_pyramid = panorama.get_laplacian_pyramid(img)
    merged_img = panorama.merge_laplacian_pyramid(laplacian_pyramid)

    plots.plot_collage(prefix + '07_laplacian.jpeg', panorama.increase_contrast(laplacian_pyramid), columns=2, rows=2)
    plots.plot_merged(prefix + '08_merged.jpeg', img, merged_img)

    result = panorama.gaussian_merge_pano(pano_image_collection, final_center_warps, output_shape)

    plots.plot_result(prefix + '09_improved_pano.jpeg', result)

    # ------------------------------------------------------------------------------------------
    # Part 7
    # ------------------------------------------------------------------------------------------

    scale = panorama.cylindrical_scales.get(i, None)
    cylindrical = panorama.warp_cylindrical(result, scale=scale)

    plots.plot_result(prefix + '10_cylindrical_pano.jpeg', cylindrical)


def main():
    if os.path.exists("./results/"):
        shutil.rmtree('./results/')
    os.makedirs('./results/')

    for i in range(5):
        try:
            process(i)
        except Exception:
            traceback.print_exc()

    s4, s5, s6, s7 = [], [], [], []
    for i in range(5):
        prefix = f"./results/{i:02d}_"
        s4 += [
            prefix + "04_simple_warps.jpeg",
            prefix + "05_final_warps.jpeg",
        ]
        s5 += [
            prefix + "06_base_pano.jpeg",
        ]
        s6 += [
            prefix + "07_laplacian.jpeg",
            prefix + "08_merged.jpeg",
            prefix + "09_improved_pano.jpeg",
        ]
        s7 += [
            prefix + "10_cylindrical_pano.jpeg",
        ]

    plots.make_ipynb(
        'results.ipynb',
        sections=[
            ("4. Преобразование всех кадров в плоскость центрального", s4),
            ("5. Совмещение всех кадров на изображении", s5),
            ("6. Склеиваем панораму с помощью пирамиды Лапласа", s6),
            ("7. Выравнивание изображений с помощью цилиндрической проекции", s7),
        ],
    )


if __name__ == '__main__':
    # Uncomment the next line to save the plots without showing them
    plots.SHOW_PLOTS = False

    main()
