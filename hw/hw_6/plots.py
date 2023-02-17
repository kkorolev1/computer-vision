import base64
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from skimage import img_as_ubyte
from skimage import io
from skimage.feature import plot_matches

FIGSIZE = (15, 10)
COLUMNS = 3
ROWS = 2
SHOW_PLOTS = True


def show_plot():
    if SHOW_PLOTS:
        plt.show()


def save_plot(filename):
    plt.savefig(filename)


def plot_collage(filename, imgs, columns=COLUMNS, rows=ROWS, figsize=FIGSIZE, title=None):
    plt.close()
    fig = plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
        plt.axis('off')

    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        try:
            img = imgs[i-1]
        except IndexError:
            break
        plt.imshow(img, interpolation='nearest')
        plt.axis('off')

    plt.tight_layout()
    save_plot(filename)
    show_plot()


def plot_keypoints(filename, img, keypoints):
    plt.close()
    plt.figure(figsize=FIGSIZE)
    plt.imshow(img)
    plt.axis('off')
    plt.scatter(keypoints[:, 1], keypoints[:, 0], facecolors='none', edgecolors='r')
    plt.tight_layout()
    save_plot(filename)
    show_plot()


def plot_inliers(filename, src, dest, src_keypoints, dest_keypoints, matches):
    plt.close()
    plt.figure(figsize=FIGSIZE)
    ax = plt.axes()
    ax.axis("off")
    ax.set_title(f"Inlier correspondences: {len(matches)} points matched")
    plot_matches(ax, src, dest, src_keypoints, dest_keypoints,
                 matches)
    plt.tight_layout()
    save_plot(filename)
    show_plot()


def plot_warps(filename, corners, output_shape=None, min_coords=None, max_coords=None, img=None):
    plt.close()
    np.random.seed(0)
    plt.figure(figsize=(15, 5))
    ax = plt.axes()

    for coords in corners:
        ax.add_patch(Polygon(coords, closed=True, fill=False, color=np.random.rand(3)))

    if max_coords is not None:
        plt.xlim(min_coords[0], max_coords[0])
        plt.ylim(max_coords[1], min_coords[1])

    if output_shape is not None:
        plt.xlim(0, output_shape[1])
        plt.ylim(output_shape[0], 0)

    if img is not None:
        plt.imshow(img)

    plt.title('Border visualization')
    plt.tight_layout()
    save_plot(filename)
    show_plot()


def plot_merged(filename, img, merged_img):
    plt.close()
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.title('Input image')
    plt.axis('off')
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.title('Merged image')
    plt.axis('off')
    plt.imshow(merged_img)
    plt.tight_layout()
    save_plot(filename)
    show_plot()


def plot_result(filename, result):
    plt.close()
    plt.figure(figsize=FIGSIZE)
    plt.imshow(result)
    plt.axis('off')
    io.imsave(filename, img_as_ubyte(result))
    plt.tight_layout()
    show_plot()


def make_ipynb(filename, sections):
    cells = []

    baseline = dict(
        cell_type='markdown',
        id='cell',
        metadata={},
    )
    index = 0

    for section, images in sections:
        header = dict(baseline)
        header["source"] = [f"# {section}\n"]
        header["id"] += str(index)
        index += 1
        cells.append(header)

        for image in images:
            stem = image.split("/")[-1]
            interlinked = dict(baseline)
            interlinked["source"] = [f"### `{stem}`\n"]
            interlinked["attachments"] = {}
            interlinked["id"] += str(index)
            index += 1

            if os.path.exists(image):
                with open(image, "rb") as fp:
                    child = base64.b64encode(fp.read()).decode()
                interlinked["attachments"][stem] = {"image/jpeg": child}
                interlinked["source"] += [f"![{stem}](attachment:{stem})\n"]
            else:
                interlinked["source"] += ["missing\n"]

            cells.append(interlinked)

    ipynb = dict(
        cells=cells,
        metadata={},
        nbformat=4,
        nbformat_minor=5,
    )
    with open(filename, "w") as fp:
        json.dump(ipynb, fp)

    # We're done.
    # You can pick up your bonus.
