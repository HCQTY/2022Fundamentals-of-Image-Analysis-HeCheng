import math
import cv2
import matplotlib.pyplot as plt






image_path = "Practice4_Test/"
image_name = "Test"
image = cv2.imread(image_path + image_name + '.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)     # F
img_h = image.shape[0]
img_w = image.shape[1]
plt.figure(1)
Original_Gray = "Original_Image_Gray"
plt.xlabel(Original_Gray)
plt.gray()
plt.imshow(image)
cv2.imwrite(image_path + Original_Gray + ".jpg", image)

# OSTU
OSTU_image = image.copy()
Sigma = -1
T = 0

for t in range(0, 256):
    bg = OSTU_image[OSTU_image <= t]
    obj = OSTU_image[OSTU_image > t]

    p0 = bg.size / OSTU_image.size
    p1 = obj.size / OSTU_image.size

    m0 = 0 if bg.size == 0 else bg.mean()
    m1 = 0 if obj.size == 0 else obj.mean()

    sigma = p0 * p1 * (m0 - m1) ** 2

    if sigma > Sigma:
        Sigma = sigma
        T = t
T = int(T)
print("OSTU threshold: {}".format(T))
plt.figure(2)
OSTU = "OSTU"
for i in range(img_h):
    for j in range(img_w):
        if(OSTU_image[i, j] >= T):
            OSTU_image[i, j] = 255
        else:
            OSTU_image[i, j] = 0
plt.imshow(OSTU_image)
cv2.imwrite(image_path + OSTU + ".jpg", OSTU_image)

# Iterative Method
Iterative_image = image.copy()
T = Iterative_image.mean()

while True:
    t0 = Iterative_image[Iterative_image < T].mean()
    t1 = Iterative_image[Iterative_image >= T].mean()
    t = (t0 + t1) / 2
    if abs(T - t) < 1:
        break
    T = t
T = int(T)
print("Iterative Method threshold: {}".format(T))
plt.figure(3)
Iterative = "Iterative"
for i in range(img_h):
    for j in range(img_w):
        if(Iterative_image[i, j] >= T):
            Iterative_image[i, j] = 255
        else:
            Iterative_image[i, j] = 0
plt.imshow(Iterative_image)
cv2.imwrite(image_path + Iterative + ".jpg", Iterative_image)


plt.show()

