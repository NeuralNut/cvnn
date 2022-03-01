import numpy as np
import pywt
from scipy import fft
from scipy.signal import spectrogram


def ts_to_cplx_scaleogram(signal,
                          scales=np.arange(1, 32),
                          wavelet='cmor1.5-1.0',
                          dt=1,
                          mode='complex'):
    if mode == 'complex':
        [X, _] = pywt.cwt(signal, scales, wavelet, dt)

    elif mode == 'split_complex':
        [X, _] = pywt.cwt(signal, scales, wavelet, dt)
        X = X.squeeze()
        X = np.expand_dims(X, axis=-1)
        X = np.concatenate((X.real, X.imag), axis=-1)

    # TODO: real wavelet feature
    elif mode == 'real':
        raise NotImplementedError

    return X


def ts_to_cplx_ts(ts, mode):
    if mode == 'complex':
        X = ts.astype(np.complex)

    elif mode == 'split_complex':
        X = ts.astype(np.complex)
        X = X.squeeze()
        X = np.expand_dims(X, axis=-1)
        X = np.concatenate((X.real, X.imag), axis=-1)

    elif mode == 'real':
        X = ts

    return X


def ts_to_cplx_spec(ts, mode):
    if mode == 'complex':
        _, _, X = spectrogram(ts, fs=1.0, return_onesided=False, nperseg=int(np.sqrt(len(ts))), axis=-1, mode='complex')

    elif mode == 'split_complex':
        _, _, X = spectrogram(ts, fs=1.0, return_onesided=False, nperseg=int(np.sqrt(len(ts))), axis=-1, mode='complex')
        X = X.squeeze()
        X = np.expand_dims(X, axis=-1)
        X = np.concatenate((X.real, X.imag), axis=-1)

    elif mode == 'real':
        _, _, X = spectrogram(ts, fs=1.0, return_onesided=True, axis=-1, mode='complex')

    return X


def ts_to_cplx_fft(ts, mode):
    if mode == 'complex':
        X = fft(ts, norm="ortho")

    elif mode == 'split_complex':
        X = fft(ts, norm="ortho")
        X = X.squeeze()
        X = np.expand_dims(X, axis=-1)
        X = np.concatenate((X.real, X.imag), axis=-1)

    elif mode == 'real':
        X = np.abs(fft(ts, norm="ortho"))

    return X