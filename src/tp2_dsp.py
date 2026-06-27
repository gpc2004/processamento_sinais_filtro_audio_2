import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import ShortTimeFFT
import soundfile as sf

try:
    import sounddevice as sd
    TEM_SOM = True
except Exception:
    TEM_SOM = False

DIR_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_SAIDA = os.path.join(DIR_BASE, "resultados_tp2")

os.makedirs(DIR_SAIDA, exist_ok=True)

#? --------------------------------------------------------------------------
#? Utilidades
#? --------------------------------------------------------------------------
def salva_figura(fig, name):
    # Monta o caminho completo do arquivo
    path = os.path.join(DIR_SAIDA, name)

    # Salva a figura no caminho definido
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Figura salva em \'{path}\'")


def toca_audio(x, fs, label=""):
    # Reproduz o audio no sistema de som, se disponivel
    if not TEM_SOM:
        print(f"Aviso: biblioteca sounddevice indisponível.  Pulando reprodução ({label}).")
        return
    
    print(f"Reproduzindo: {label}...")

    # X é o vetor de amostras do áudio
    # Normaliza o áudio antes de tocar, dividindo o sinal inteiro pelo maior valor absoluto
    xn = x / (np.max(np.abs(x)) + 1e-12) * 0.9 # 0.9 p/ evitar risco de saturação

    # Toca o aúdio
    sd.play(xn.astype(np.float32), fs)
    sd.wait() # Espera reprodução do áudio acabar


def gerar_sinal_sintetico(fs=44100, duracao=10.0, semente=42):
    # Gera tom + ruído de faixa larga em trecho central
    # para permitir testar o código sem o arquivo original (audio_corrompido.wav)
    n = int(fs * duracao) # num total de amostras
    t = np.arange(n) / fs
    x = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 1200 * t) # Sinal limpo

    r0, r1 = int(0.35 * n), int(0.65 * n) # Intervalo com ruído
    rng = np.random.default_rng(semente)
    ruido = rng.standard_normal(n)

    # Ruido de faixa larga 5-18kHz, aproximadamente
    # S0S = Second-Order Sections -> mais estável do que representar filtros de ordem alta diretamente por polinômios.
    sos = signal.butter(6, [5000, 18000], btype="bandpass", fs=fs, output="sos") # filtro Butterworth ordem 6 passa-faixa
    ruido_filtrado = signal.sosfilt(sos, ruido)

    # Aplica os ruídos no intervalo
    mask = np.zeros(n)
    mask[r0:r1] = 1.0

    x = x + 0.6 * ruido_filtrado * mask
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.9

    # Retorna o sinal sintético em formato de número real de 64 bits
    return x.astype(np.float64), fs


#* --------------------------------------------------------------------------
#* 1.1 - Carregamento e visualização do sinal corrompido
#* --------------------------------------------------------------------------
def carregar_e_plotar(path):
    # Verifica se o sinal existe
    if path and os.path.isfile(path):
        # Lê o áudio e vetoriza
        x, fs = sf.read(path)  

        # Tratamento para áudios com mais de 1 canal
        if x.ndim > 1:
            #  Pega a média
            x = x.mean(axis=1)

        print(f"Arquivo carregado: {path} | fs = {fs} Hz | N = {len(x)} amostras")

    else:
        print("Arquivo de audio não encontrado - usando sinal sintético de teste.")
        x, fs = gerar_sinal_sintetico()

    n = len(x)
    t = np.arange(n) / fs

    # Cálculo do espectro
    X = np.fft.fftshift(np.fft.fft(x))
    f = np.fft.fftshift(np.fft.fftfreq(n, d=1 / fs)) / 1000.0  # kHz

    # Gráfico
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(t, x)
    axs[0].set_xlabel("t (s)")
    axs[0].set_ylabel("x(t)")
    axs[0].set_title("Sinal corrompido - domínio do tempo")

    axs[1].plot(f, np.abs(X) / n)
    axs[1].set_xlabel("f (kHz)")
    axs[1].set_ylabel("|X(e^{j$\\omega$})|")
    axs[1].set_title("Espectro de amplitude")
    fig.tight_layout()
    salva_figura(fig, "1_sinal_corrompido.png")
    plt.close(fig)

    # Retorna vetor de amostras do sinal de áudio + fs
    return x, fs

#* --------------------------------------------------------------------------
#* 1.2 Na main
#* --------------------------------------------------------------------------

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
    salva_figura(fig, "2_janela_kaiser.png")
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
    salva_figura(fig, "3_resposta_impulso_fir.png")
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
    salva_figura(fig, "4_resposta_freq_fir.png")
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
    salva_figura(fig, "5_sinal_filtrado_tempo.png")
    plt.close(fig)

    Y = np.fft.fftshift(np.fft.fft(y_de))
    f = np.fft.fftshift(np.fft.fftfreq(n, d=1 / fs)) / 1000.0
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(f, np.abs(Y) / n)
    ax.set_xlabel("f (kHz)")
    ax.set_ylabel("|Y(e^{j$\\omega$})|")
    ax.set_title("Espectro do sinal filtrado (FIR)")
    fig.tight_layout()
    salva_figura(fig, "6_sinal_filtrado_freq.png")
    plt.close(fig)

    return y_de  # usaremos a versao "equacao de diferencas" como referencia

def execucao():
    if len(sys.argv) > 1:
        wav_path = sys.argv[1]
    else:
        wav_path = os.path.join(DIR_BASE, "data", "audio_corrompido.wav")

    #! 1.1 - Carregamento e plot
    x, fs = carregar_e_plotar(wav_path)

    #! 1.2 - Reprodução do sinal corrompido
    toca_audio(x, fs, label="sinal corrompido")


if __name__ == "__main__":
    execucao()