from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import math

# =========================================
# ĐỌC FILE WAV
# =========================================

fs, x = wavfile.read('CV01_noisy_2000Hz_tone.wav')

# Nếu stereo -> lấy 1 kênh
if len(x.shape) > 1:
    x = x[:,0]

# Chuyển sang float
x = x.astype(float)

# Lấy 3 giây đầu
x = x[:fs*3]

# Trục thời gian
t = np.arange(len(x)) / fs

# =========================================
# WAVEFORM TRƯỚC LỌC
# =========================================

plt.figure(figsize=(12,4))

plt.plot(t, x)

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.title('Waveform Before Filtering')

plt.grid()

plt.tight_layout()

plt.savefig('waveform_before.png', dpi=300)

plt.show()

# =========================================
# FFT TRƯỚC LỌC
# =========================================

N = len(x)

X = np.fft.fft(x)

freq = np.fft.fftfreq(N, 1/fs)

plt.figure(figsize=(12,4))

plt.plot(freq[:N//2], np.abs(X[:N//2]))

plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.title('FFT Before Filtering')

plt.grid()

plt.tight_layout()

plt.savefig('fft_before.png', dpi=300)

plt.show()

# =========================================
# FIR NOTCH FILTER
# =========================================

# Tần số nhiễu
f0 = 2000

# Tần số góc chuẩn hóa
w0 = 2 * np.pi * f0 / fs

# Tín hiệu sau lọc
y = np.zeros(len(x))

# FIR Notch
for n in range(2, len(x)):
    y[n] = x[n] - 2*np.cos(w0)*x[n-1] + x[n-2]

# =========================================
# WAVEFORM SAU LỌC
# =========================================

plt.figure(figsize=(12,4))

plt.plot(t, y)

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.title('Waveform After FIR Notch Filter')

plt.grid()

plt.tight_layout()

plt.savefig('waveform_after.png', dpi=300)

plt.show()

# =========================================
# FFT SAU LỌC
# =========================================

Y = np.fft.fft(y)

plt.figure(figsize=(12,4))

plt.plot(freq[:N//2], np.abs(Y[:N//2]))

plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.title('FFT After FIR Notch Filter')

plt.grid()

plt.tight_layout()

plt.savefig('fft_after.png', dpi=300)

plt.show()

# =========================================
# TÍNH NĂNG LƯỢNG
# =========================================

# Dải tần chứa nhiễu
band = (freq >= 1900) & (freq <= 2100)

# Năng lượng trước lọc
energy_before = np.sum(np.abs(X[band])**2)

# Năng lượng sau lọc
energy_after = np.sum(np.abs(Y[band])**2)

# =========================================
# TÍNH TỶ LỆ GIẢM NHIỄU
# =========================================

reduction = (energy_before - energy_after) / energy_before * 100

# =========================================
# IN KẾT QUẢ
# =========================================

print("=================================")

exp = int(math.floor(math.log10(abs(energy_before))))
mantissa = energy_before / 10**exp

print(f"Energy Before Filtering = {mantissa:.2f} × 10^{exp}")

exp2 = int(math.floor(math.log10(abs(energy_after))))
mantissa2 = energy_after / 10**exp2

print(f"Energy After Filtering = {mantissa2:.2f} × 10^{exp2}")

print("Noise Reduction = ", reduction, "%")

print("=================================")