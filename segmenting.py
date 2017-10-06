import os, sys
import numpy as np
#import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from scipy.misc import imsave

from skimage import io
from skimage.filters import sobel, threshold_local
from skimage.morphology import watershed


def open_image(name):
    #filename = os.path.join(os.getcwd(), name)
    filename = os.path.isdir(name) and name or os.path.join(os.getcwd(), name)
    return io.imread(filename, as_grey=True)


def adaptive_threshold(image, BLOCK_SIZE=701):
    # Create threshold image
    # Offset is not desirable for these images
    threshold_img = threshold_local(image, BLOCK_SIZE)

    # Binarize the image with the threshold image
    binary_adaptive = image < threshold_img

    # Convert the mask (which has dtype bool) to dtype int
    # This is required for the code in `segmentize` (below) to work
    return binary_adaptive.astype(int)


def segmentize(image, FG=1, BG=2):
    # make segmentation using edge-detection and watershed
    edges = sobel(image)
    markers = np.zeros_like(image)

    markers[image == 0] = BG
    markers[image == 1] = FG

    ws = watershed(edges, markers)
    return ndi.label(ws == FG)


def find_segment(segments, index):
    seg = np.where(segments == index)
    shape = segments.shape

    minx, maxx = max(seg[0].min() - 1, 0), min(seg[0].max() + 1, shape[0])
    miny, maxy = max(seg[1].min() - 1, 0), min(seg[1].max() + 1, shape[1])

    im = segments[minx:maxx, miny:maxy] == index
    return np.sum(im), np.invert(im)


def run(f):
    print('Processing:', f)
    image = open_image(f)
    processed = adaptive_threshold(image)

    segments = segmentize(processed)
    print('Segments detected:', segments[1] - 1)

    segs = [ find_segment(segments[0], s) for s in range(1, segments[1]) ]
    segs.sort(key=lambda s: -s[0])

    # Get the directory name (if a full path is given)
    #folder = r'C:\Users\yourname\Desktop\sketch2\data'
    folder = os.getcwd()

    # Get the file name
    filenm = os.path.basename(f)[:-4]

    # If it doesn't already exist, create a new dir "segments" to save the PNGs
    segments_folder = os.path.join(folder, filenm + "_segments")
    os.path.isfile(segments_folder) and os.remove(segments_folder)
    os.path.isdir(segments_folder)  or  os.mkdir(segments_folder)

    # Save the segments to the "segments" directory
    r3 = list(range(3))
    i = 0
    sep = '-'
    ext = '.png'

    for s in segs:
        # Create an MxNx4 array (RGBA)
        s1 = s[1]
        seg_rgba = np.zeros((s1.shape[0], s1.shape[1], 4), dtype=np.bool)

        # Fill R, G and B with copies of the image
        for c in r3: seg_rgba[:,:,c] = s1

        # For A (alpha), use invert of image (so background is 0=transparent)
        seg_rgba[:,:,3] = ~s1

        # Save image
        fullpath = os.path.join(segments_folder, filenm + sep + str(i) + ext)
        #print(fullpath)
        imsave(fullpath, seg_rgba)
        i += 1


for f in sys.argv[1:]: run(f)
