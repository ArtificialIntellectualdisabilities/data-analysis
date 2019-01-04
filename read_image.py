import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
i=mpimg.imread('D:\\fengxu\\bee.png',format='rb+')
print(i.shape)
plt.imshow(i)
plt.show()
