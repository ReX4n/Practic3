import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog as fd
import tkinter.messagebox as mb
import operator
from skimage import io, transform, metrics


def gradient(image):

    window_size = 3
    dx = 1
    dy = 1

    # calculate gradient
    gradient_x = cv2.Sobel(image, cv2.CV_32F, dx, 0, ksize=window_size)
    gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, dy, ksize=window_size)

    # calculate absolute gradient
    abs_gradient_x = cv2.convertScaleAbs(gradient_x)
    abs_gradient_y = cv2.convertScaleAbs(gradient_y)

    # calculate result gradient
    result_gradient = cv2.addWeighted(abs_gradient_x, 0.5, abs_gradient_y, 0.5, 0)

    sum_grad = []
    for i in range(0, len(result_gradient), 1):
        sum_grad.append(round(sum(result_gradient[i]) / len(result_gradient[i]), 1))
    return sum_grad


def dct(file):
    # Application of two-dimensional discrete cosine transformation (DCT)
    dct_result = cv2.dct(np.float32(file))
    return dct_result


def dft(file):

    # Application of two-dimensional discrete Fourier transform (DFT)
    dft_result = cv2.dft(np.float32(file), flags=cv2.DFT_COMPLEX_OUTPUT)

    # Zero frequency shift to the center
    dft_shift = np.fft.fftshift(dft_result)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    return magnitude_spectrum


def histogram(file):
    # reading the input image
    # computing the histogram of the blue channel of the image
    histogram_result = cv2.calcHist([file], [0], None, [256], [0, 256])
    return histogram_result


def scale(file):
    image = io.imread(file)
    scale_resolution = 20
    image_result = transform.resize(image, (scale_resolution, scale_resolution))
    return image_result


def plot_graphics(count_standards):
    statistic_dct = []
    statistic_dft = []
    statistic_scale = []
    statistic_histogram = []
    statistic_gradient = []
    delta_coefficient_histogram = 190
    delta_coefficient_gradient = 77
    test_img_array = []
    test_histogram = []
    test_gradient = []
    test_dft = []
    test_dct = []
    test_scale = []
    standard_img_array = []
    standard_histogram = []
    standard_gradient = []
    standard_dft = []
    standard_dct = []
    standard_scale = []

    for i in range(1, 11):
        sum_histogram = 0
        sum_gradient = 0
        sum_similarity_dft = 0
        sum_similarity_dct = 0
        sum_similarity_scale = 0
        for j in range(1, count_standards + 1):
            result_histogram = 0
            result_gradient = 0
            standard_img = cv2.imread(f"s{i}/{j}.pgm", cv2.IMREAD_GRAYSCALE)
            standard_img_array.append(standard_img)
            standard_histogram.append(histogram(standard_img))
            standard_gradient.append(gradient(standard_img))
            standard_dft.append(dft(standard_img))
            standard_dct.append(dct(standard_img))
            standard_scale.append(scale(f"s{i}/{j}.pgm"))

            for k in range(count_standards + 1, 11):
                test_img = cv2.imread(f"s{i}/{k}.pgm", cv2.IMREAD_GRAYSCALE)
                test_img_array.append(test_img)
                test_histogram.append(histogram(test_img))
                test_gradient.append(gradient(test_img))
                test_dft.append(dft(test_img))
                test_dct.append(dct(test_img))
                test_scale.append(scale(f"s{i}/{k}.pgm"))

                index_standard_histogram, standard_max_histogram =\
                    max(enumerate(standard_histogram[j-1+count_standards*(i-1)]), key=operator.itemgetter(1))
                index_standard_gradient, standard_max_gradient =\
                    max(enumerate(standard_gradient[j-1+count_standards*(i-1)]), key=operator.itemgetter(1))

                test_max_histogram =\
                    test_histogram[k - count_standards - 1+count_standards*(i-1)][index_standard_histogram]
                test_max_gradient =\
                    test_gradient[k - count_standards - 1+count_standards*(i-1)][index_standard_gradient]

                delta_histogram = abs(standard_max_histogram-test_max_histogram)
                delta_gradient = abs(standard_max_gradient-test_max_gradient)

                if delta_histogram < delta_coefficient_histogram:
                    result_histogram += 1
                if delta_gradient < delta_coefficient_gradient:
                    result_gradient += 1
                mean_mag_standard = np.mean(standard_dft[j-1+count_standards*(i-1)])
                mean_mag_test = np.mean(test_dft[k-count_standards-1+count_standards*(i-1)])
                similarity_percent_dft = mean_mag_test / mean_mag_standard
                if similarity_percent_dft > 1:
                    similarity_percent_dft = 2-similarity_percent_dft
                sum_similarity_dft += similarity_percent_dft
                linalg_norm_standard = np.linalg.norm(standard_dct[j-1+count_standards*(i-1)])
                linalg_norm_test = np.linalg.norm(test_dct[k-count_standards-1+count_standards*(i-1)])
                similarity_percent_dct = linalg_norm_test/linalg_norm_standard
                if similarity_percent_dct > 1:
                    similarity_percent_dct = 2 - similarity_percent_dct
                sum_similarity_dct += similarity_percent_dct
                ssim = metrics.structural_similarity(standard_scale[j-1+count_standards*(i-1)],
                                                     test_scale[k-count_standards-1+(10-count_standards)*(i-1)],
                                                     data_range=255)
                if ssim > 1:
                    ssim = 2 - ssim
                sum_similarity_scale += ssim
            sum_histogram += result_histogram
            sum_gradient += result_gradient
        statistic_histogram.append((sum_histogram/((10-count_standards)*count_standards)))
        statistic_gradient.append(sum_gradient / ((10-count_standards)*count_standards))
        statistic_dft.append(sum_similarity_dft / ((10-count_standards)*count_standards))
        statistic_dct.append(sum_similarity_dct / ((10-count_standards)*count_standards))
        statistic_scale.append(sum_similarity_scale / ((10-count_standards)*count_standards))

    print(statistic_histogram)
    print(statistic_gradient)
    print(statistic_dft)
    print(statistic_dct)
    print(statistic_scale)
    fig1, ((ax1, ax2, ax3, ax4, ax5, ax6), (ax11, ax22, ax33, ax44, ax55, ax66)) = plt.subplots(2, 6)
    fig2, (axH, axG, axDf, axDc, axScale) = plt.subplots(1, 5)
    plt.ion()
    image_a = ax11.imshow(test_img_array[0])
    histogram_a, = ax22.plot(test_histogram[0])
    dft_a = ax33.imshow(test_dft[0])
    dct_a = ax44.imshow(np.abs(test_dct[0]))
    gradient_a, = ax55.plot(np.arange(len(test_gradient[0])), test_gradient[0])
    scale_a = ax66.imshow(test_scale[0])
    image_b = ax1.imshow(standard_img_array[0])
    histogram_b, = ax2.plot(standard_histogram[0])
    dft_b = ax3.imshow(standard_dft[0])
    dct_b = ax4.imshow(np.abs(standard_dct[0]))
    gradient_b, = ax5.plot(np.arange(len(standard_gradient[0])), standard_gradient[0])
    scale_b = ax6.imshow(standard_scale[0])

    axH.plot(np.arange(len(statistic_histogram)), statistic_histogram)
    axH.set_title("Histogram")
    axG.plot(np.arange(len(statistic_gradient)), statistic_gradient)
    axG.set_title("Gradient")
    axDf.plot(np.arange(len(statistic_dft)), statistic_dft)
    axDf.set_title("DFT")
    axDc.plot(np.arange(len(statistic_dct)), statistic_dct)
    axDc.set_title("DCT")
    axScale.plot(np.arange(len(statistic_scale)), statistic_scale)
    axScale.set_title("Scale")
    fig2.show()

    fig1.show()

    for t in range(0, 10):
        for p in range(count_standards*t, count_standards*t+count_standards):
            image_b.set_data(standard_img_array[p])
            histogram_b.set_ydata(standard_histogram[p])
            dft_b.set_data(standard_dft[p])
            dct_b.set_data(standard_dct[p])
            gradient_b.set_ydata(standard_gradient[p])
            scale_b.set_data(standard_scale[p])
            fig1.canvas.draw()
            fig1.canvas.flush_events()
            for m in range(p*(10-count_standards), (10-count_standards)*(p+1)):
                image_a.set_data(test_img_array[m])
                histogram_a.set_ydata(test_histogram[m])
                dft_a.set_data(test_dft[m])
                dct_a.set_data(test_dct[m])
                gradient_a.set_ydata(test_gradient[m])
                scale_a.set_data(test_scale[m])
                fig1.canvas.draw()
                fig1.canvas.flush_events()


def plot_graphics_chosen(filename1, filename2):
    delta_coefficient_histogram = 100
    delta_coefficient_gradient = 100
    result_histogram = 0
    result_gradient = 0
    standard_img = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
    standard_histogram = histogram(standard_img)
    standard_gradient = gradient(standard_img)
    standard_dft = dft(standard_img)
    standard_dct = dct(standard_img)
    standard_scale = scale(filename1)
    test_img = cv2.imread(filename2, cv2.IMREAD_GRAYSCALE)
    test_histogram = histogram(test_img)
    test_gradient = gradient(test_img)
    test_dft = dft(test_img)
    test_dct = dct(test_img)
    test_scale = scale(filename2)
    test_or_img = plt.imread(filename2, cv2.IMREAD_GRAYSCALE)
    standard_or_img = plt.imread(filename1, cv2.IMREAD_GRAYSCALE)
    index_standard_histogram, standard_max_histogram = max(enumerate(standard_histogram), key=operator.itemgetter(1))
    index_standard_gradient, standard_max_gradient = max(enumerate(standard_gradient), key=operator.itemgetter(1))

    test_max_histogram = test_histogram[index_standard_histogram]
    test_max_gradient = test_gradient[index_standard_gradient]

    delta_histogram = abs(standard_max_histogram-test_max_histogram)
    delta_gradient = abs(standard_max_gradient-test_max_gradient)

    if delta_histogram < delta_coefficient_histogram:
        result_histogram += 1
    if delta_gradient < delta_coefficient_gradient:
        result_gradient += 1
    plt.ion()
    plt.subplot(3, 6, 13)
    plt.imshow(standard_or_img)
    plt.subplot(3, 6, 14)
    plt.plot(standard_histogram)
    plt.subplot(3, 6, 15)
    plt.imshow(standard_dft)
    plt.subplot(3, 6, 16)
    plt.imshow(np.abs(standard_dct))
    plt.subplot(3, 6, 17)
    plt.plot(np.arange(len(standard_gradient)), standard_gradient)
    plt.subplot(3, 6, 18)
    plt.imshow(standard_scale)

    plt.subplot(3, 6, 1)
    plt.imshow(test_or_img)
    plt.subplot(3, 6, 2)
    plt.plot(test_histogram)
    plt.subplot(3, 6, 3)
    plt.imshow(test_dft)
    plt.subplot(3, 6, 4)
    plt.imshow(np.abs(test_dct))
    plt.subplot(3, 6, 5)
    plt.plot(np.arange(len(test_gradient)), test_gradient)
    plt.subplot(3, 6, 6)
    plt.imshow(test_scale)

    if result_gradient != 0 and result_histogram != 0:
        show_res("Совпадает")
    else:
        show_res("Не совпадает")
    plt.show()


def get_count_standards():
    count_standards = num_standards_entry.get()
    if count_standards.isdigit() and int(count_standards) > 0:
        plot_graphics(int(count_standards))
    else:
        tk.showerror("Ошибка", "Введите целое положительное число")


def choose_test():
    filename1 = fd.askopenfilename()
    filename2 = fd.askopenfilename()
    plot_graphics_chosen(filename1, filename2)


def show_res(text):
    mb.showinfo("Результат", text)


# Create main window
root = tk.Tk()
num_standards_label = tk.Label(root, text="Количество эталонов:")
num_standards_label.pack()
num_standards_entry = tk.Entry(root)
num_standards_entry.pack()

plot_button = tk.Button(root, text="Построить графики", command=get_count_standards)
plot_button.pack()
plot_button = tk.Button(root, text="Произвести произвольную выборку", command=choose_test)
plot_button.pack()

root.mainloop()
