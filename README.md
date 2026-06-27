# Processamento e Restauração de Sinal de Áudio - TP2

Este repositório contém a implementação do segundo trabalho prático da disciplina **Processamento de Sinais (ELE042 - UFMG)**.

O projeto dá continuidade ao TP1, substituindo a abordagem com filtro IIR por um filtro FIR passa-baixas projetado pelo método da janela de Kaiser. Também é incluída uma etapa bônus de filtragem adaptativa baseada em STFT.

## Objetivos

* Analisar o áudio corrompido nos domínios do tempo e da frequência.
* Projetar um filtro FIR passa-baixas por janela de Kaiser.
* Aplicar a filtragem por três métodos:

  * equação de diferenças;
  * convolução direta;
  * domínio da frequência.
* Comparar os resultados obtidos com a abordagem IIR do TP1.
* Implementar uma filtragem adaptativa via STFT como bônus.
* Gerar automaticamente gráficos e arquivos de áudio filtrados.

## Estrutura do repositório

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── tp2_dsp.py
├── data/
│   └── audio_corrompido.wav
├── resultados_tp2/
└── relatorio/
    └── Relatorio_Trabalho_Pratico_2.pdf
```

## Dependências

As bibliotecas necessárias estão listadas em `requirements.txt`.

Para instalar:

```bash
pip install -r requirements.txt
```

## Como executar

A partir da pasta principal do repositório:

```bash
python src/tp2_dsp.py
```

Por padrão, o programa utiliza o arquivo:
```text
data/audio_corrompido.wav
```

Os gráficos e áudios gerados são salvos automaticamente na pasta:

```text
resultados_tp2/
```

Também é possível passar outro arquivo de áudio como parâmetro. Para isso, informe o caminho do arquivo na chamada do programa:
```bash
python src/tp2_dsp.py caminho/para/outro_audio.wav
```

Por exemplo, caso exista outro arquivo dentro da pasta data, chamado meu_audio.wav, execute:
```bash
python src/tp2_dsp.py data/meu_audio.wav
```

Nesse caso, o código processará o áudio informado em vez de utilizar o arquivo padrão data/audio_corrompido.wav.

## Arquivos principais

* `src/tp2_dsp.py`: script principal do trabalho.
* `data/audio_corrompido.wav`: áudio de entrada utilizado no processamento.
* `resultados_tp2/`: pasta com os resultados gerados.
* `relatorio/`: pasta destinada ao relatório final em PDF.

## Observação

Os arquivos `.mat` utilizados no TP1 não são necessários neste trabalho, pois os coeficientes do filtro FIR são calculados diretamente pelo código.
