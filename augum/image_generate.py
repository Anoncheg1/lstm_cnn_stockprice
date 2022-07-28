from augum.da305 import main
import numpy as np

batches, w_size, fut_l = main()
print("image_gen", batches.shape, w_size, fut_l)
print(batches[0][0])

# for window, _ in batches:  # window/future
#     data = np.zeros((h, w, 3), dtype=np.uint8)
#     data = np.zeros((h, w, 3), dtype=np.uint8)
#     for pv in window:


