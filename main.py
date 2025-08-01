"""
Where our works begin
"""
import numpy as np

from L1_minimization import *
from util import *

if __name__ == "__main__"  :
    # read grayscale image
    image = cv.imread("smallTriangle.png", cv.IMREAD_GRAYSCALE)

    show(image)

    result = do_Fourier_transform(image)

    missing = np.complex128(0 + 1j)

    print(result[0])

    result[0][2] = missing

    print(result[0])

    result_hat = do_inverse_Fourier_transform(result)

    show(result_hat)









