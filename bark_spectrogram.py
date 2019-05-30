import numpy as np
import librosa
import librosa.core

def hz2bark(f):
    """
    Convert frequencies (Hertz) to Bark frequencies

    :param f: the input frequency
    :return:
    """
    return 6. * np.arcsinh(f / 600.)



def bark2hz(z):
    """
    Converts frequencies Bark to Hertz (Hz)

    :param z:
    :return:
    """
    return 600. * np.sinh(z / 6.)

def fft2barkmx(n_fft, fs, nfilts=128, width=1., minfreq=0., maxfreq=8000):
    """
    Generate a matrix of weights to combine FFT bins into Bark
    bins.  n_fft defines the source FFT size at sampling rate fs.
    Optional nfilts specifies the number of output bands required
    (else one per bark), and width is the constant width of each
    band in Bark (default 1).
    While wts has n_fft columns, the second half are all zero.
    Hence, Bark spectrum is fft2barkmx(n_fft,fs) * abs(fft(xincols, n_fft));
    2004-09-05  dpwe@ee.columbia.edu  based on rastamat/audspec.m

    :param n_fft: the source FFT size at sampling rate fs
    :param fs: sampling rate
    :param nfilts: number of output bands required. n_mels for melspectrogram
    :param width: constant width of each band in Bark (default 1)
    :param minfreq:
    :param maxfreq:
    :return: a matrix of weights to combine FFT bins into Bark bins
    """

    maxfreq = max(maxfreq, fs / 2.)

    min_bark = hz2bark(minfreq)
    nyqbark = hz2bark(maxfreq) - min_bark

    if nfilts == 0:
        nfilts = np.ceil(nyqbark) + 1

    wts = np.zeros((nfilts, int(1+ n_fft//2)))

    # bark per filt
    step_barks = nyqbark / (nfilts - 1)

    # Frequency of each FFT bin in Bark
    binbarks = hz2bark(np.arange(n_fft / 2 + 1) * fs / n_fft)

    for i in range(nfilts):
        f_bark_mid = min_bark + i * step_barks
        # Linear slopes in log-space (i.e. dB) intersect to trapezoidal window
        lof = (binbarks - f_bark_mid - 0.5)
        hif = (binbarks - f_bark_mid + 0.5)
        wts[i, :n_fft // 2 + 1] = 10 ** (np.minimum(np.zeros_like(hif), np.minimum(hif, -2.5 * lof) / width))

    return wts

