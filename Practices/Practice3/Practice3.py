import math
import cv2
import matplotlib.pyplot as plt

def DPCM(yBuf, dBuf, reBuf, w, h, bitnum):
    limit = math.pow(2, bitnum - 1)

    for i in range(h):
        tmp = 0
        # tmp = yBuf[i][0]
        for j in range(w):
            out = int(yBuf[i][j]) - tmp
            if out < -limit:
                out = -limit
            elif out > (limit-1):
                out = limit-1
            dBuf[i][j] = out

            reBuf[i][j] = out + tmp
            tmp = int(reBuf[i][j])



image_path = "Practice3_Test/"
image_name = "Test"
image = cv2.imread(image_path + image_name + '.jpg')
yBuf = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)     # F
img_h = yBuf.shape[0]
img_w = yBuf.shape[1]
plt.figure(1)
Original_Gray = "Original_Image_Gray"
plt.xlabel(Original_Gray)
plt.gray()
plt.imshow(yBuf)
cv2.imwrite(image_path + Original_Gray + ".jpg", yBuf)


dBuf = yBuf.copy()
reBuf = yBuf.copy()


fig2 = plt.figure(2)
plt.title("bitnum = 8")
DPCM(yBuf, dBuf, reBuf, img_w, img_h, 8)
# plt.subplot(1, 3, 1)
ax1 = plt.subplot(1, 3, 1)
ax1.imshow(yBuf)
ax2 = plt.subplot(1, 3, 2)
ax2.imshow(dBuf)
ax3 = plt.subplot(1, 3, 3)
ax3.imshow(reBuf)


fig3 = plt.figure(3)
plt.title("bitnum = 4")
DPCM(yBuf, dBuf, reBuf, img_w, img_h, 4)
ax1 = fig3.add_subplot(1, 3, 1)
ax1.imshow(yBuf)
ax2 = fig3.add_subplot(1, 3, 2)
ax2.imshow(dBuf)
ax3 = fig3.add_subplot(1, 3, 3)
ax3.imshow(reBuf)

fig4 = plt.figure(4)
plt.title("bitnum = 2")
DPCM(yBuf, dBuf, reBuf, img_w, img_h, 2)
ax1 = fig4.add_subplot(1, 3, 1)
ax1.imshow(yBuf)
ax2 = fig4.add_subplot(1, 3, 2)
ax2.imshow(dBuf)
ax3 = fig4.add_subplot(1, 3, 3)
ax3.imshow(reBuf)

fig5 = plt.figure(5)
plt.title("bitnum = 1")
DPCM(yBuf, dBuf, reBuf, img_w, img_h, 1)
ax1 = fig5.add_subplot(1, 3, 1)
ax1.imshow(yBuf)
ax2 = fig5.add_subplot(1, 3, 2)
ax2.imshow(dBuf)
ax3 = fig5.add_subplot(1, 3, 3)
ax3.imshow(reBuf)


plt.show()
