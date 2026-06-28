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
    fig.savefig(path, dpi=300, bbox_inches="tight")
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
    limite_f = np.zeros(n)
    limite_f[r0:r1] = 1.0

    x = x + 0.6 * ruido_filtrado * limite_f
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
    axs[1].set_ylabel(r"$|X(e^{j\omega})|$")
    axs[1].set_title("Espectro de amplitude")
    fig.tight_layout()
    salva_figura(fig, "1_sinal_corrompido.png")
    plt.close(fig)

    # Retorna vetor de amostras do sinal de áudio + fs
    return x, fs


#* --------------------------------------------------------------------------
#* 1.2 No main
#* --------------------------------------------------------------------------


#* --------------------------------------------------------------------------
#* 1.3 - Projeto do filtro FIR (janela de Kaiser)
#* --------------------------------------------------------------------------
def design_fir_kaiser(fs, fp=5000.0, fr=6000.0, ripple_porcentagem=0.1):
    # Projeta filtro FIR passa-baixas pelo metodo da janela de Kaiser
    delta = ripple_porcentagem / 100.0  # 0.1% -> 0.001
    atenuacao_db = -20 * np.log10(delta)  # atenuação equivalente em dB

    largura_hz = fr - fp

    # Estima os parâmetros necessários para projetar um filtro FIR com janela de Kaiser
    num_coef, beta = signal.kaiserord(atenuacao_db, largura_hz / (fs / 2))
    if num_coef % 2 == 0:  #  Se for par
        num_coef += 1  # ordem N (número de coeficientes) ímpar -> fase linear tipo I

    # Frequência de corte é a média
    fc = (fp + fr) / 2.0

    # Projeta o filtro FIR pelo método da janela
    h = signal.firwin(num_coef, fc, window=("kaiser", beta), fs=fs)

    print(f"--- Projeto FIR (Kaiser) ---")
    print(f"Atenuação alvo (Ap = {ripple_porcentagem}%): {atenuacao_db:.2f} dB")
    print(f"Ordem do filtro N (número de coeficientes): {num_coef}")
    print(f"Parâmetro beta: {beta:.4f}")
    print(f"Frequência de corte (fc): {fc:.1f} Hz")

    w_kaiser = signal.windows.kaiser(num_coef, beta) # Vetor com num_coef amostras

    # Gráfico da janela de Kaiser usada
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(num_coef), w_kaiser, marker=".", ms=3)
    ax.set_xlabel("n")
    ax.set_ylabel("w[n]")
    ax.set_title(f"Janela de Kaiser (N={num_coef}, beta={beta:.3f})")
    fig.tight_layout()
    salva_figura(fig, "2_janela_kaiser.png")
    plt.close(fig)

    return h, num_coef, beta


#* --------------------------------------------------------------------------
#* 1.4 - Resposta ao impulso e respostas de magnitude
#* --------------------------------------------------------------------------
def plota_respostas_filtro(h, fs, fmax_khz=22):
    # Tamanho do vetor
    n = len(h)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    markerline, stemlines, baseline = ax.stem(np.arange(n), h, basefmt=" ")
    plt.setp(markerline, markersize=3)
    plt.setp(stemlines, linewidth=0.8)
    ax.set_xlabel("n")
    ax.set_ylabel(r"$h[n]$")
    ax.set_title("Resposta ao impulso do filtro FIR")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    salva_figura(fig, "3_resposta_impulso_fir.png")
    plt.close(fig)

    # Cálculo da resposta em frequência
    w, H = signal.freqz(h, worN=8192, fs=fs)
    f_khz = w / 1000.0 # Converte o eixo de frequências de Hz para kHz
    limite_f = f_khz <= fmax_khz # Limite para mostrar penas frequências até fmax_khz

    # Plot com magnitude e fase
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(f_khz[limite_f], 20 * np.log10(np.abs(H[limite_f]) + 1e-12))
    axs[0].set_ylabel(r"$|X(e^{j\omega})|$ (dB)")
    axs[0].set_title("Resposta em magnitude")
    axs[0].grid(True)

    axs[1].plot(f_khz[limite_f], np.unwrap(np.angle(H[limite_f])))
    axs[1].set_xlabel("f (kHz)")
    axs[1].set_ylabel(r"$\theta(\omega)$ (rad)")
    axs[1].set_title("Resposta em fase")
    axs[1].grid(True)
    fig.tight_layout()
    salva_figura(fig, "4_resposta_freq_fir.png")
    plt.close(fig)


#* --------------------------------------------------------------------------
#* 1.5 - Filtragem por 3 métodos (equação de diferenças, convolução e frequência)
#* --------------------------------------------------------------------------
def filtra_equacao_diff(x, b):
    # Filtragem via equação de diferenças
    # FIR: a=[1], H(z) = B(z) / 1
    return signal.lfilter(b, [1.0], x)


def filtra_convolucao(x, h):
    # Filtragem via convolução direta (linear, full, depois truncada).
    y = np.convolve(x, h, mode="full")
    return y[: len(x)]  # mesmo comprimento de x para compa direta


def filtra_frequencia(x, h):
    # Filtragem no domínio da frequência via FFT (convolução circular
    # com zero-padding suficiente para evitar wrap-around -> equivalente
    # a convolução linear).
    n = len(x) + len(h) - 1
    nfft = 1
    while nfft < n:
        nfft *= 2
    X = np.fft.fft(x, nfft)
    H = np.fft.fft(h, nfft)
    y = np.real(np.fft.ifft(X * H))
    return y[: len(x)]


def comparacao_filtros(x, h, fs):
    y_eq = filtra_equacao_diff(x, h)
    y_conv = filtra_convolucao(x, h)
    y_freq = filtra_frequencia(x, h)

    err_conv = np.max(np.abs(y_eq - y_conv))
    err_freq = np.max(np.abs(y_eq - y_freq))
    print(f"Erro máximo |dif.eq - convolução| = {err_conv:.3e}")
    print(f"Erro máximo |dif.eq - frequência|  = {err_freq:.3e}")

    # Plots
    n = len(x)
    t = np.arange(n) / fs
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, y_eq, label="Equação de diferenças")
    ax.plot(t, y_conv, "--", label="Convolução", alpha=0.7)
    ax.plot(t, y_freq, ":", label="Domínio da frequência", alpha=0.7)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("y[n]")
    ax.legend()
    ax.set_title("Comparação entre métodos de implementação do filtro FIR")
    fig.tight_layout()
    salva_figura(fig, "5_sinal_filtrado_tempo.png")
    plt.close(fig)

    Y = np.fft.fftshift(np.fft.fft(y_eq))
    f = np.fft.fftshift(np.fft.fftfreq(n, d=1 / fs)) / 1000.0
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(f, np.abs(Y) / n)
    ax.set_xlabel("f (kHz)")
    ax.set_ylabel(r"$|Y(e^{j\omega})|$")
    ax.set_title("Espectro do sinal filtrado (FIR)")
    fig.tight_layout()
    salva_figura(fig, "6_sinal_filtrado_freq.png")
    plt.close(fig)

    return y_eq  # Retorna a versão da equação de diferenças como referência


#* --------------------------------------------------------------------------
#* 1.6 - No main
#* --------------------------------------------------------------------------



def execucao():
    if len(sys.argv) > 1:
        wav_path = sys.argv[1]
    else:
        wav_path = os.path.join(DIR_BASE, "data", "audio_corrompido.wav")

    #! 1.1 - Carregamento e plot
    x, fs = carregar_e_plotar(wav_path)

    #! 1.2 - Reprodução do sinal corrompido
    #toca_audio(x, fs, label="sinal corrompido")

    #! 1.3 - Projeto do filtro FIR
    h, N, beta = design_fir_kaiser(fs, fp=5000.0, fr=6000.0, ripple_porcentagem=0.1)

    #! 1.4 - Respostas do gráfico
    plota_respostas_filtro(h, fs)

    #! 1.5 - Filtragem (3 métodos) + comparação
    y_fir = comparacao_filtros(x, h, fs)

    #! 1.6 - Reprodução do sinal filtrado (linear/FIR)
    toca_audio(y_fir, fs, label="Sinal filtrado (FIR linear)")

    #! Bônus
    #todo


if __name__ == "__main__":
    execucao()