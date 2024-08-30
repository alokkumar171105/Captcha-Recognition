
# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a list of strings. Make sure each string is either "ODD"
# or "EVEN" (without the quotes) depending on whether the hexadecimal number in
# the image is odd or even. Take care not to make spelling or case mistakes. Make
# sure that the length of the list returned as output is the same as the number of
# filenames that were given as input. The judge may give unexpected results if this
# convention is not followed.

import numpy as np

import cv2

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def thin_lines(image, kernel_size):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    thinned = cv2.erode(binary, kernel, iterations=1)

    return thinned

def decaptcha(filepaths):

  num_test = len(filepaths)

  lim = 87

  y1=np.zeros(num_test, dtype = int)

  ref_filepaths = [ '/content/%d.png' % i for i in range( 16 ) ]

  ref_imgs = np.empty((16, 7), dtype=object)

  angles = [-30,-20,-10,0,10,20,30]

  for j in range(16):

    f = ref_filepaths[j]
    ref = cv2.imread(f)
    l = ref[5:95, 3:93]

    l = thin_lines(l,2)

    l = cv2.resize(l, (lim, lim))

    for i in range(7):
      a = angles[i]
      rotated = rotate_image(l, a)
      ref_imgs[j][i] = rotated

  ref_imgs = np.array(ref_imgs)

  pred = np.zeros([16,7])

  predictions = ['ODD' for n in range(num_test)]

  p = 0

  for f in filepaths:

    img = cv2.imread(f)

    k = img[10:100,360:450]

    k = thin_lines(k,2)

    k_resized = cv2.resize(k, (lim, lim))

    for i in range(16):

      for j in range(7):

        l_resized = ref_imgs[i][j]

        m = np.subtract(k_resized, l_resized)

        sum = np.sum(m[0:lim, 0:lim])
        avg = sum
        pred[i,j] = avg

    pos = np.argwhere(pred == np.min(pred))

    o=pos[0,0]

    if o == 0 or o==2 or o==4 or o==6 or o==8 or o==10 or o==12 or o==14:
      predictions[p] = 'EVEN'

    p += 1

  return predictions
