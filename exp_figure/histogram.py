from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
src=Image.open('0_image75.jpg')
r,g,b=src.split()

plt.figure("lena")
ar=np.array(r).flatten()
plt.hist(ar, bins=256, normed=1,facecolor='r',edgecolor='r',hold=1)
ag=np.array(g).flatten()
plt.hist(ag, bins=256, normed=1, facecolor='g',edgecolor='g',hold=1)
ab=np.array(b).flatten()
plt.hist(ab, bins=256, normed=1, facecolor='b',edgecolor='b')
plt.show()