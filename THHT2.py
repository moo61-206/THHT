from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

# ====================================
# ĐỌC FILE WAV
# ====================================

fs, x = wavfile.read('CV02_noisy_highfreq_hiss.wav')

# Nếu stereo -> lấy 1 kênh
if len(x.shape) > 1:
    x = x[:,0]

# Chuyển sang float
x = x.astype(float)

# Lấy 3 giây đầu để vẽ đẹp
x = x[:fs*3]

# Trục thời gian
t = np.arange(len(x)) / fs

# ====================================
# WAVEFORM TRƯỚC LỌC
# ====================================

plt.figure(figsize=(12,4))

plt.plot(t, x)

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.title('Waveform Before Moving Average Filter')

plt.grid()

plt.savefig('waveform_before_MA.png', dpi=300)

plt.show()

# ====================================
# FFT TRƯỚC LỌC
# ====================================

N = len(x)

X = np.fft.fft(x)

freq = np.fft.fftfreq(N, 1/fs)

plt.figure(figsize=(12,4))

plt.plot(freq[:N//2], np.abs(X[:N//2]))

plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.title('FFT Before Filtering')

plt.grid()

plt.savefig('fft_before_MA.png', dpi=300)

plt.show()

# ====================================
# MOVING AVERAGE FILTER
# ====================================

# Kích thước cửa sổ
M = 5

# Bộ lọc trung bình trượt
y = np.convolve(x, np.ones(M)/M, mode='same')

# ====================================
# WAVEFORM SAU LỌC
# ====================================

plt.figure(figsize=(12,4))

plt.plot(t, y)

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.title('Waveform After Moving Average Filter')

plt.grid()

plt.savefig('waveform_after_MA.png', dpi=300)

plt.show()

# ====================================
# FFT SAU LỌC
# ====================================

Y = np.fft.fft(y)

plt.figure(figsize=(12,4))

plt.plot(freq[:N//2], np.abs(Y[:N//2]))

plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.title('FFT After Filtering')

plt.grid()

plt.savefig('fft_after_MA.png', dpi=300)

plt.show()

# ====================================
# TÍNH TỶ LỆ GIẢM NHIỄU
# ====================================

# Chọn dải tần cao
band = (freq >= 4000) & (freq <= fs/2)

# Năng lượng trước lọc
before = np.sum(np.abs(X[band])**2)

# Năng lượng sau lọc
after = np.sum(np.abs(Y[band])**2)

# Tỷ lệ giảm nhiễu
reduction = (before - after) / before * 100

print("Noise Reduction = ", reduction, "%")