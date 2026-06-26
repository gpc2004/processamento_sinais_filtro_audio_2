# Processamento e Restauração de Sinal de Áudio - TP2

Este repositório contém a implementação do segundo trabalho prático da disciplina **Processamento de Sinais (ELE042 - UFMG)**.

O projeto dá continuidade ao primeiro trabalho prático, no qual foi utilizado um filtro IIR para restaurar um áudio corrompido por ruído de faixa larga. Nesta segunda etapa, o objetivo é reprojetar a filtragem utilizando um filtro FIR passa-baixas projetado pelo método da janela de Kaiser, além de implementar uma abordagem adaptativa baseada na Transformada Breve de Fourier (STFT).

## Objetivos

- Carregar e analisar o sinal de áudio corrompido nos domínios do tempo e da frequência.
- Projetar um filtro FIR passa-baixas pelo método da janela de Kaiser.
- Comparar diferentes formas de filtragem:
  - equação de diferenças;
  - convolução direta;
  - filtragem no domínio da frequência.
- Comparar qualitativamente a filtragem FIR com a abordagem IIR do primeiro trabalho.
- Implementar, como bônus, uma filtragem adaptativa baseada em STFT.
- Gerar gráficos e arquivos de áudio filtrados automaticamente.

## Estrutura do repositório

```text
.
├── README.md
├── requirements.txt
├── src/
│   └── tp2_dsp.py
├── data/
│   └── audio_corrompido.wav
├── resultados_tp2/
│   ├── figuras geradas pelo script
│   └── áudios filtrados
└── relatorio/
    └── Relatorio_Trabalho_Pratico_2.pdf
