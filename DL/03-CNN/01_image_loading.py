import matplotlib.pyplot as plt
import numpy as np

img1 = np.zeros([200, 200, 3])
plt.imshow(img1)
plt.show()

img2 = np.full([200, 200, 3], 128)
plt.imshow(img2)
plt.show()

img2 = np.full([200, 200, 3], 256)
plt.imshow(img2)
plt.show()

# img = plt.imread('./data/img.jpg')
# print(img.shape)
# plt.imshow(img)
# plt.show()


img = plt.imread('./data/img.jpg')
print(img.shape)
plt.imshow(img)
plt.show()