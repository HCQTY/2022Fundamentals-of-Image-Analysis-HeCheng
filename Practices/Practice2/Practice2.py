import matplotlib.pyplot as graph
import numpy as np
from numpy import fft
import math
import cv2


# 仿真运动模糊
def motion_process(image_size, motion_angle):
    PSF = np.zeros(image_size)
    center_position = (image_size[0] - 1) / 2

    slope_tan = math.tan(motion_angle * math.pi / 180)
    slope_cot = 1 / slope_tan
    if slope_tan <= 1:
        for i in range(20):
            offset = round(i * slope_tan)
            PSF[int(center_position + offset), int(center_position - offset)] = 1
        return PSF / PSF.sum()  # 对点扩散函数进行归一化亮度
    else:
        for i in range(20):
            offset = round(i * slope_cot)
            PSF[int(center_position - offset), int(center_position + offset)] = 1
        return PSF / PSF.sum()


# 对图片进行运动模糊
def make_blurred(input, PSF, eps):
    input_fft = fft.fft2(input)  # 进行二维数组的傅里叶变换
    PSF_fft = fft.fft2(PSF) + eps
    blurred = fft.ifft2(input_fft * PSF_fft)
    blurred = np.abs(fft.fftshift(blurred))
    return blurred

def wiener_with_AF(input, PSF, N, F, gamma=0.01):

    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF)
    PSF_fft_conj = np.conj(PSF_fft)
    N_fft = fft.fft2(N)
    F_fft = fft.fft2(F)
    S_n = np.conj(N_fft) * N_fft
    S_f = np.conj(F_fft) * F_fft
    PSF_fft_1 = PSF_fft_conj / (PSF_fft_conj * PSF_fft + gamma * S_n / S_f)
    result = fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(fft.fftshift(result))

    return result

def wiener_without_SNR(input, PSF, K=0.01):

    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF)
    PSF_fft_conj = np.conj(PSF_fft)
    PSF_fft_1 = PSF_fft_conj * PSF_fft / (PSF_fft_conj * PSF_fft + K) / PSF_fft
    result = fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(fft.fftshift(result))

    return result

def wiener_with_SNR(input, PSF, SNR, gamma=0.01):

    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF)
    PSF_fft_conj = np.conj(PSF_fft)
    PSF_fft_1 = PSF_fft_conj * PSF_fft / (PSF_fft_conj * PSF_fft + gamma * SNR) / PSF_fft
    result = fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(fft.fftshift(result))

    return result


image_path = "Practice2_Test/"
image_name = "Test"
image = cv2.imread(image_path + image_name + '.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)     # F
img_h = image.shape[0]
img_w = image.shape[1]
graph.figure(1)
Original_Gray = "Original_Image_Gray"
graph.xlabel(Original_Gray)
graph.gray()
graph.imshow(image)
cv2.imwrite(image_path + Original_Gray + ".jpg", image)

graph.figure(2)
graph.gray()
PSF = motion_process((img_h, img_w), 60)
blurred = np.abs(make_blurred(image, PSF, 1e-3))        # 进行运动模糊处理
Motion_Blurred = "Motion_Blurred"
graph.xlabel(Motion_Blurred)
graph.imshow(blurred)
cv2.imwrite(image_path + Motion_Blurred + ".jpg", blurred)

graph.figure(3)
N = 0.1 * blurred.std() * np.random.standard_normal(blurred.shape)  # 添加噪声,standard_normal产生随机的函数
blurred_noise = blurred + N
Motion_Noise_Blurred = "Motion_Noise_Blurred"
graph.xlabel(Motion_Noise_Blurred)
graph.imshow(blurred_noise)  # 显示添加噪声且运动模糊的图像
cv2.imwrite(image_path + Motion_Noise_Blurred + ".jpg", blurred_noise)

graph.figure(4)
result = wiener_without_SNR(blurred_noise, PSF, 0.01)  # 维纳滤波信噪比未知
Wiener_Deblurred_Without_SNR = "Wiener_Deblurred_Without_SNR"
graph.xlabel(Wiener_Deblurred_Without_SNR)
graph.imshow(result)
cv2.imwrite(image_path + Wiener_Deblurred_Without_SNR + ".jpg", result)

graph.figure(5)
SNR = (N ** 2).mean()
print(SNR)
result = wiener_with_SNR(blurred_noise, PSF, SNR, 0.07)  # 维纳滤波信噪比已知
Wiener_Deblurred_With_SNR = "Wiener_Deblurred_With_SNR"
graph.xlabel(Wiener_Deblurred_With_SNR)
graph.imshow(result)
cv2.imwrite(image_path + Wiener_Deblurred_With_SNR + ".jpg", result)

graph.figure(6)
result = wiener_with_AF(blurred, PSF, N, image, 0.01)  # 维纳滤波自相关函数已知
Wiener_Deblurred_With_AF = "Wiener_Deblurred_With_AF"
graph.xlabel(Wiener_Deblurred_With_AF)
graph.imshow(result)
cv2.imwrite(image_path + Wiener_Deblurred_With_AF + ".jpg", result)

graph.show()

