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

# Análise comparativa simples
def medir_energia(y, fc, fs):
    # Calcula uma métrica simples de energia espectral acima de uma frequência de corte
    Y = np.abs(np.fft.rfft(y))

    # Cria o eixo de frequências correspondente a FFT real do sinal  
    f = np.fft.rfftfreq(len(y), d=1 / fs)

    # Seleciona apenas as frequências acima de fc e soma o quadrado das magnitudes
    return np.sum(Y[f >= fc] ** 2)


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

#? --------------------------------------------------------------------------
#? BÔNUS - 2.1 STFT com reconstrução perfeita
#? --------------------------------------------------------------------------
def calcula_stft(fs, tamanho_janela=1024, deslocamento_janela=4):
    # Constroi um objeto ShortTimeFFT com janela de Hann e parametros que
    # satisfazem a condição de reconstrução perfeita).

    # Calcula o deslocamento entre uma janela e a próxima.
    janela = tamanho_janela // deslocamento_janela  # 75% de sobreposição

    janela_hann = signal.windows.hann(tamanho_janela, sym=False)

    # Objeto usado para calcular a STFT
    SFT = ShortTimeFFT(janela_hann, hop=janela, fs=fs, mfft=tamanho_janela, scale_to="magnitude")

    print(f"--- STFT (bônus) ---")
    print(f"Janela: Hann\nLargura = {tamanho_janela} amostras\nPasso (janela) = {janela} amostras")
    print(f"Sobreposição: {100 * (1 - janela / tamanho_janela):.0f}%")

    return SFT


#? --------------------------------------------------------------------------
#? BÔNUS - 2.2 Espectograma
#? --------------------------------------------------------------------------
def plota_espectrograma(SFT, x, fs):
    # Calcula a STFT do sinal x
    Sx = SFT.stft(x)

    # Eixo de tempo
    t_stft = SFT.t(len(x))
    # Eixo da frequência
    f_stft = SFT.f / 1000.0  # kHz

    mag_db = 20 * np.log10(np.abs(Sx) + 1e-12) # Evita erro com log 0

    # Plota 2D
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.pcolormesh(t_stft, f_stft, mag_db, shading="gouraud", cmap="viridis")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("f (kHz)")
    ax.set_title("Espectrograma (STFT) - 2D")
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    salva_figura(fig, "7_espectrograma_2d.png")
    plt.close(fig)

    # Plota 3D
    fig = plt.figure(figsize=(9, 6))
    ax3 = fig.add_subplot(111, projection="3d")
    Tg, Fg = np.meshgrid(t_stft, f_stft)
    step = max(1, mag_db.shape[1] // 200)  # reduz pontos p/ performance
    ax3.plot_surface(Tg[:, ::step], Fg[:, ::step], mag_db[:, ::step], cmap="viridis", linewidth=0)
    ax3.set_xlabel("t (s)")
    ax3.set_ylabel("f (kHz)")
    ax3.set_zlabel("dB")
    ax3.set_title("Espectrograma (STFT) - 3D")
    fig.tight_layout()
    salva_figura(fig, "8_espectrograma_3d.png")
    plt.close(fig)

    return Sx

#? --------------------------------------------------------------------------
#? BÔNUS - 2.3 Filtragem adaptativa
#? --------------------------------------------------------------------------
def filtro_adaptativo(x, fs, h, SFT, fr=6000.0, limiar=None):
    """
    Para cada coluna (frame) da STFT:
      - Calcula a variância das magnitudes acima de fr (6 kHz);
      - Se a variância for alta (indício de ruído de faixa larga),
        usa, para aquele frame, a STFT do sinal já filtrado pelo FIR
        (calculado via equacao de diferencas no dominio do tempo);
      - Caso contrário, mantem a STFT do sinal original (preserva altas
        frequências do áudio limpo).
    Em seguida, reconstroi via iSTFT.
    """

    # Sinal totalmente filtrado pelo FIR (equação de diferenças) - candidato
    y_filt_full = filtra_equacao_diff(x, h)
    # Garante mesmo comprimento para o objeto STFT
    n = len(x)
    y_filt_full = y_filt_full[:n]

    Sx_orig = SFT.stft(x)
    Sx_filt = SFT.stft(y_filt_full)

    # Obtém vetor de frequêmncias da STFT em Hz
    f_stft = SFT.f
    # Seleciona os índices das frequências maiores ou iguais a 6 kHz
    indices_af = np.where(f_stft >= fr)[0]

    # Calcula a variância das magnitudes acima de 6kHz
    variacia_por_frame = np.var(np.abs(Sx_orig[indices_af, :]), axis=0)

    if limiar is None:
        # Limiar adaptativo: média + 1 desvio padrão da variância entre os frames
        limiar = np.mean(variacia_por_frame) + np.std(variacia_por_frame)

    # Máscara booleana que classifica, para cada frame:
    # True: o frame foi classificado como ruidoso, então será usada a versão filtrada
    # False: o frame foi classificado como limpo, então será mantida a versão original
    mascara_decisao = variacia_por_frame > limiar

    print(f"--- Filtragem adaptativa ---")
    print(f"Limiar de variância usado: {limiar:.3e}")
    print(f"Frames filtrados: {np.sum(mascara_decisao)} / {len(mascara_decisao)}")

    # Construção da STFT híbrida
    Sx_hybrid = Sx_orig.copy()
    # Para todos os frames classificados como ruidosos, substituímos a STFT original pela STFT do sinal filtrado pelo FIR
    Sx_hybrid[:, mascara_decisao] = Sx_filt[:, mascara_decisao]

    # Reconstrução por iSTFT
    y_adapt = SFT.istft(Sx_hybrid, k1=n)
    y_adapt = np.real(y_adapt[:n])

    # Gráfico da decisão de filtragem ao longo do tempo
    t_frames = SFT.t(n)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(t_frames, variacia_por_frame, label="Variância (>6kHz) por frame")
    ax.axhline(limiar, color="r", ls="--", label="Limiar")
    ax.fill_between(t_frames, 0, variacia_por_frame.max(), where=mascara_decisao,
                     color="red", alpha=0.15, label="Filtro FIR aplicado")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Variância")
    ax.legend()
    ax.set_title("Decisão de filtragem adaptativa por frame (STFT)")
    fig.tight_layout()
    salva_figura(fig, "9_decisao_filtragem_adaptativa.png")
    plt.close(fig)

    # Retornamos o sinal filtrado adaptativamente, a máscara indicando quais frames foram filtrados e a variância calculada por frame
    return y_adapt, mascara_decisao, variacia_por_frame


#? --------------------------------------------------------------------------
#? BÔNUS - 2.4 Reconstrução do sinal filtrado
#? --------------------------------------------------------------------------
def plota_sinal_tempo_e_freq(y, fs, title, fname):
    # Número de amostras do sinal
    n = len(y)

    # Eixo do tempo
    t = np.arange(n) / fs

    # Eixo da frequência (em kHz)
    f = np.fft.fftshift(np.fft.fftfreq(n, d=1 / fs)) / 1000.0

    # FFT do sinal
    Y = np.fft.fftshift(np.fft.fft(y))

    # Plota
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(t, y)
    axs[0].set_xlabel("t (s)")
    axs[0].set_title(f"{title} - tempo")
    axs[1].plot(f, np.abs(Y) / n)
    axs[1].set_xlabel("f (kHz)")
    axs[1].set_title(f"{title} - frequencia")
    fig.tight_layout()
    salva_figura(fig, fname)
    plt.close(fig)


def execucao():
    if len(sys.argv) > 1:
        wav_path = sys.argv[1]
    else:
        wav_path = os.path.join(DIR_BASE, "data", "audio_corrompido.wav")

    #! 1.1 - Carregamento e plot
    x, fs = carregar_e_plotar(wav_path)

    #! 1.2 - Reprodução do sinal corrompido
    toca_audio(x, fs, label="sinal corrompido")

    #! 1.3 - Projeto do filtro FIR
    h, N, beta = design_fir_kaiser(fs, fp=5000.0, fr=6000.0, ripple_porcentagem=0.1)

    #! 1.4 - Respostas do gráfico
    plota_respostas_filtro(h, fs)

    #! 1.5 - Filtragem (3 métodos) + comparação
    y_fir = comparacao_filtros(x, h, fs)

    #! 1.6 - Reprodução do sinal filtrado (linear/FIR)
    toca_audio(y_fir, fs, label="Sinal filtrado (FIR linear)")

    #! 1.7 - Como depende do TP1, foi realizada no relatório

    #? Bônus
    #! 2.1 - Calcula a STFT do sinal
    SFT = calcula_stft(fs, tamanho_janela=1024, deslocamento_janela=4)

    #! 2.2 - Plota espectrograma
    plota_espectrograma(SFT, x, fs)

    #! 2.3 - Realiza a filtragem adaptativa do sinal
    y_adapt, mask, var_frames = filtro_adaptativo(x, fs, h, SFT, fr=6000.0)

    #! 2.4 - Reconstrução do sinal filtrado utilizando a iSTFT
    plota_sinal_tempo_e_freq(y_adapt, fs, "Sinal filtrado (adaptativo)", "10_sinal_adaptativo.png")

    #! 2.5 - Reprodução do sinal filtrado 
    toca_audio(y_adapt, fs, label="sinal filtrado (adaptativo - STFT)")

    #! 2.6 - Análises comparativas
    e_orig_hf = medir_energia(x, 6000, fs)
    e_fir_hf = medir_energia(y_fir, 6000, fs)
    e_adapt_hf = medir_energia(y_adapt, 6000, fs)

    print("\n--- Resumo comparativo (energia espectral acima de 6 kHz) ---")
    print(f"Sinal original         : {e_orig_hf:.3e}")
    print(f"Filtrado FIR (linear)  : {e_fir_hf:.3e}  ({100*e_fir_hf/e_orig_hf:.1f}% do original)")
    print(f"Filtrado adaptativo    : {e_adapt_hf:.3e}  ({100*e_adapt_hf/e_orig_hf:.1f}% do original)")
    print("\n=> O filtro linear remove componentes de alta frequencia em TODO o sinal\n"
          "   (inclusive onde não há ruído), 'abafando' o áudio limpo.\n"
          "=> O filtro adaptativo preserva as altas frequências nos trechos sem ruído\n"
          "   detectado, aplicando a atenuação apenas onde a variância espectral em\n"
          "   alta frequência indica a presenca do ruído de faixa larga.")

    # Salva os áudios resultantes
    sf.write(os.path.join(DIR_SAIDA, "saida_fir.wav"), y_fir / (np.max(np.abs(y_fir)) + 1e-9), fs)
    sf.write(os.path.join(DIR_SAIDA, "saida_adaptativa.wav"), y_adapt / (np.max(np.abs(y_adapt)) + 1e-9), fs)
    print(f"\nTodas as figuras e áudios de saida foram salvos em: {os.path.abspath(DIR_SAIDA)}")

if __name__ == "__main__":
    execucao()