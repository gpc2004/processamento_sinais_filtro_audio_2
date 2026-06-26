# -*- coding: utf-8 -*-
"""
TP2 - Processamento de Sinais (ELE042 - UFMG)
Filtragem FIR (janela de Kaiser) + Filtragem adaptativa via STFT (bonus)

Autor: (preencher nomes do grupo)

Como usar:
    python tp2_dsp.py audio_corrompido.wav

Se nenhum arquivo for passado, ou o arquivo nao existir, o script gera
um sinal sintetico de teste (tom + ruido de faixa larga em um trecho)
para que toda a pipeline possa ser demonstrada mesmo sem o .wav original.

Dependencias:
    pip install numpy scipy matplotlib soundfile sounddevice
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import ShortTimeFFT

import soundfile as sf

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:
    HAS_SOUNDDEVICE = False

OUT_DIR = "resultados_tp2"
os.makedirs(OUT_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[fig salva] {path}")


def play_audio(x, fs, label=""):
    """Reproduz o audio no sistema de som, se disponivel."""
    if not HAS_SOUNDDEVICE:
        print(f"[aviso] sounddevice indisponivel - pulando reproducao ({label}).")
        return
    print(f"Reproduzindo: {label} ...")
    xn = x / (np.max(np.abs(x)) + 1e-12) * 0.9
    sd.play(xn.astype(np.float32), fs)
    sd.wait()


def generate_synthetic_signal(fs=44100, dur=10.0):
    """Gera sinal sintetico (tom + ruido de faixa larga em trecho central)
    para permitir testar o script sem o arquivo original."""
    n = int(fs * dur)
    t = np.arange(n) / fs
    x = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 1200 * t)

    i0, i1 = int(0.35 * n), int(0.65 * n)
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(n)
    # ruido de faixa larga 5-18kHz aproximadamente
    sos = signal.butter(6, [5000, 18000], btype="bandpass", fs=fs, output="sos")
    noise_bp = signal.sosfilt(sos, noise)
    mask = np.zeros(n)
    mask[i0:i1] = 1.0
    x = x + 0.6 * noise_bp * mask
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.9
    return x.astype(np.float64), fs


# --------------------------------------------------------------------------
# 1.1 - Carregamento e visualizacao do sinal corrompido
# --------------------------------------------------------------------------
def load_and_plot_signal(path):
    if path and os.path.isfile(path):
        x, fs = sf.read(path)
        if x.ndim > 1:
            x = x.mean(axis=1)
        print(f"Arquivo carregado: {path} | fs = {fs} Hz | N = {len(x)} amostras")
    else:
        print("[aviso] Arquivo de audio nao encontrado - usando sinal sintetico de teste.")
        x, fs = generate_synthetic_signal()

    n = len(x)
    t = np.arange(n) / fs

    X = np.fft.fftshift(np.fft.fft(x))
    f = np.fft.fftshift(np.fft.fftfreq(n, d=1 / fs)) / 1000.0  # kHz

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(t, x)
    axs[0].set_xlabel("t (s)")
    axs[0].set_ylabel("x(t)")
    axs[0].set_title("Sinal corrompido - dominio do tempo")

    axs[1].plot(f, np.abs(X) / n)
    axs[1].set_xlabel("f (kHz)")
    axs[1].set_ylabel("|X(e^{j$\\omega$})|")
    axs[1].set_title("Espectro de amplitude")
    fig.tight_layout()
    savefig(fig, "1_sinal_corrompido.png")
    plt.close(fig)

    return x, fs


# --------------------------------------------------------------------------
# 1.3/1.4 - Projeto do filtro FIR (janela de Kaiser)
# --------------------------------------------------------------------------
def design_fir_kaiser(fs, fp=5000.0, fr=6000.0, ripple_pct=0.1):
    """Projeta filtro FIR passa-baixas pelo metodo da janela de Kaiser.
    Ap = Ar = ripple_pct (%) -> convertido para atenuacao em dB.
    """
    delta = ripple_pct / 100.0  # 0.1% -> 0.001
    atten_db = -20 * np.log10(delta)  # atenuacao equivalente (~60 dB para 0.1%)

    width_hz = fr - fp
    numtaps, beta = signal.kaiserord(atten_db, width_hz / (fs / 2))
    if numtaps % 2 == 0:
        numtaps += 1  # ordem N (numero de coeficientes) impar -> fase linear tipo I

    fc = (fp + fr) / 2.0
    h = signal.firwin(numtaps, fc, window=("kaiser", beta), fs=fs)

    print(f"--- Projeto FIR (Kaiser) ---")
    print(f"Atenuacao alvo (Ap=Ar={ripple_pct}%): {atten_db:.2f} dB")
    print(f"Ordem do filtro N (numero de coeficientes): {numtaps}")
    print(f"Parametro beta: {beta:.4f}")
    print(f"Frequencia de corte (fc): {fc:.1f} Hz")

    # Grafico da janela de Kaiser usada
    w_kaiser = signal.windows.kaiser(numtaps, beta)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(numtaps), w_kaiser, marker=".", ms=3)
    ax.set_xlabel("n")
    ax.set_ylabel("w[n]")
    ax.set_title(f"Janela de Kaiser (N={numtaps}, beta={beta:.3f})")
    fig.tight_layout()
    savefig(fig, "2_janela_kaiser.png")
    plt.close(fig)

    return h, numtaps, beta


def plot_filter_response(h, fs, fmax_khz=22):
    n = len(h)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.stem(np.arange(n), h, basefmt=" ")
    ax.set_xlabel("n")
    ax.set_ylabel("h[n]")
    ax.set_title("Resposta ao impulso do filtro FIR")
    fig.tight_layout()
    savefig(fig, "3_resposta_impulso_fir.png")
    plt.close(fig)

    w, H = signal.freqz(h, worN=8192, fs=fs)
    f_khz = w / 1000.0
    mask = f_khz <= fmax_khz

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(f_khz[mask], 20 * np.log10(np.abs(H[mask]) + 1e-12))
    axs[0].set_ylabel("|H(e^{j$\\omega$})| (dB)")
    axs[0].set_title("Resposta em magnitude")
    axs[0].grid(True)

    axs[1].plot(f_khz[mask], np.unwrap(np.angle(H[mask])))
    axs[1].set_xlabel("f (kHz)")
    axs[1].set_ylabel("$\\theta(\\omega)$ (rad)")
    axs[1].set_title("Resposta em fase")
    axs[1].grid(True)
    fig.tight_layout()
    savefig(fig, "4_resposta_freq_fir.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# 1.5 - Filtragem por 3 metodos (equacao de diferencas, convolucao, frequencia)
# --------------------------------------------------------------------------
def filter_diff_equation(x, b):
    """Filtragem via equacao de diferencas (FIR: a=[1])."""
    return signal.lfilter(b, [1.0], x)


def filter_convolution(x, h):
    """Filtragem via convolucao direta (linear, full, depois truncada)."""
    y = np.convolve(x, h, mode="full")
    return y[: len(x)]  # mesmo comprimento de x para comparacao direta


def filter_frequency_domain(x, h):
    """Filtragem no dominio da frequencia via FFT (convolucao circular
    com zero-padding suficiente para evitar wrap-around -> equivalente
    a convolucao linear)."""
    n = len(x) + len(h) - 1
    nfft = 1
    while nfft < n:
        nfft *= 2
    X = np.fft.fft(x, nfft)
    H = np.fft.fft(h, nfft)
    y = np.real(np.fft.ifft(X * H))
    return y[: len(x)]


def compare_filtering_methods(x, h, fs):
    y_de = filter_diff_equation(x, h)
    y_conv = filter_convolution(x, h)
    y_freq = filter_frequency_domain(x, h)

    err_conv = np.max(np.abs(y_de - y_conv))
    err_freq = np.max(np.abs(y_de - y_freq))
    print(f"Erro maximo |dif.eq - convolucao| = {err_conv:.3e}")
    print(f"Erro maximo |dif.eq - frequencia|  = {err_freq:.3e}")

    n = len(x)
    t = np.arange(n) / fs
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, y_de, label="Equacao de diferencas")
    ax.plot(t, y_conv, "--", label="Convolucao", alpha=0.7)
    ax.plot(t, y_freq, ":", label="Dominio da frequencia", alpha=0.7)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("y[n]")
    ax.legend()
    ax.set_title("Sinal filtrado (FIR) - comparacao dos metodos")
    fig.tight_layout()
    savefig(fig, "5_sinal_filtrado_tempo.png")
    plt.close(fig)

    Y = np.fft.fftshift(np.fft.fft(y_de))
    f = np.fft.fftshift(np.fft.fftfreq(n, d=1 / fs)) / 1000.0
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(f, np.abs(Y) / n)
    ax.set_xlabel("f (kHz)")
    ax.set_ylabel("|Y(e^{j$\\omega$})|")
    ax.set_title("Espectro do sinal filtrado (FIR)")
    fig.tight_layout()
    savefig(fig, "6_sinal_filtrado_freq.png")
    plt.close(fig)

    return y_de  # usaremos a versao "equacao de diferencas" como referencia


