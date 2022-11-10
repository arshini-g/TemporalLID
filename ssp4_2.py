import numpy as np
import librosa as lb
import librosa.display as lbd
import scipy as sp
import matplotlib.pyplot as plt

y, sr = lb.load('test.wav')
y = y[23000:28120]
yf = lb.util.frame(y, frame_length=1024, hop_length=256)


yf = yf.T
y_predicted = np.zeros(yf.shape)
for i in range(yf.shape[0]):
    a = lb.lpc(yf[i], order=9)
    a = -a[1:]
    a = np.insert(a,0,0)
    y_predicted[i] = sp.signal.lfilter(a, [1], yf[i])


y_res = []
y_res = y_predicted[0]
for i in range(1,y_predicted.shape[0]): # Combining back the frames
    y_res = np.append(y_res,y_predicted[i][768:])


plt.subplot(2,2,1)
plt.plot(y)
plt.ylabel("Amplitude")
plt.title('Original signal')

plt.subplot(2,2,2)
plt.plot(y_res)
plt.ylabel("Amplitude")
plt.title('Predicted signal')

plt.subplot(2,2,3)
plt.plot(y-y_res)
plt.ylabel("Amplitude")
plt.title('Error Signal')

plt.subplot(2,2,4)  
plt.stem(a)
plt.ylabel("Amplitude")
plt.title('LP Coefficients of last frame')

plt.savefig('lp.png')
plt.show()

plt.plot(y)
plt.plot(y_res,linestyle='--')
plt.title('LP Prediction')
plt.legend(['Original', 'Predicted'])
plt.savefig('lp2.png')
plt.show()