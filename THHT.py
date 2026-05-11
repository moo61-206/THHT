from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

# ====================================
# ĐỌC FILE WAV
# ====================================

fs, x = wavfile.read('CV00_reference_clean_synthetic_voice.wav')

# Nếu file stereo -> lấy 1 kênh
if len(x.shape) > 1:
    x = x[:,0]

# Chuyển dữ liệu sang float
x = x.astype(float)

# ====================================
# LẤY 3 GIÂY ĐẦU
# ====================================

x = x[:fs*3]

# ====================================
# TẠO TRỤC THỜI GIAN
# ====================================

t = np.arange(len(x)) / fs

# ====================================
# VẼ WAVEFORM
# ====================================

plt.figure(figsize=(12,4))

plt.plot(t, x)

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.title('Waveform of Audio Signal')

plt.grid()

plt.tight_layout()

# ====================================
# LƯU ẢNH
# ====================================

plt.savefig('waveform.png', dpi=300)

# ====================================
# HIỂN THỊ ĐỒ THỊ
# ====================================

plt.show()