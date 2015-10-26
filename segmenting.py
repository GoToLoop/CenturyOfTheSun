import os, sys
import numpy as np
from scipy import ndimage as ndi
from scipy.misc import imsave
import matplotlib.pyplot as plt

from skimage.filters import sobel, threshold_adaptive
from skimage.morphology import watershed
from skimage import io


def open_image(name):
    filename = os.path.join(os.getcwd(), name)
    return io.imread(filename)


def adaptive_threshold(image):
    block_size = 40
    binary_adaptive = threshold_adaptive(image, block_size, offset=10)
    return np.invert(binary_adaptive) * 1.


def segmentize(image):
    # make segmentation using edge-detection and watershed
    edges = sobel(image)
    markers = np.zeros_like(image)
    foreground, background = 1, 2
    markers[image == 0] = background
    markers[image == 1] = foreground

    ws = watershed(edges, markers)

    return ndi.label(ws == foreground)


def find_segment(segments, index):
    segment = np.where(segments == index)
    shape = segments.shape

    minx, maxx = max(segment[0].min() - 1, 0), min(segment[0].max() + 1, shape[0])
    miny, maxy = max(segment[1].min() - 1, 0), min(segment[1].max() + 1, shape[1])

    im = segments[minx:maxx, miny:maxy] == index

    return (np.sum(im), np.invert(im))


def run(f):
    print('Processing:', f)

    image = open_image(f)
    processed = adaptive_threshold(image)
    segments = segmentize(processed)

    print('Segments detected:', segments[1])

    seg = []
    for s in range(1, segments[1]):
        seg.append(find_segment(segments[0], s))

    seg.sort(key=lambda s: -s[0])

    for i in range(len(seg)):
        imsave('segments/' + f + '_' + str(i) + '.png', seg[i][1])

os.mkdir(os.path.join(os.getcwd(), 'segments'))
for f in sys.argv[1:]:
    run(f)
