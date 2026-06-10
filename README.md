# Progetto IMAD A.A 2025/2026

Questa repository contiene il progetto svolto per il corso **IMAD**  A.A. 2025/2026 – Università degli Studi di Pavia.

L’obiettivo del progetto è la **modellazione e previsione del carico elettrico** a partire da dati meteorologici (temperature) e informazioni temporali, confrontando diversi approcci di regressione lineare e non lineare.


## Descrizione delle cartelle

### `data_analysis`
Contiene gli script per l’**analisi esplorativa dei dati**, tra cui:
- visualizzazione del dataset
- suddivisione train/validation/test
- analisi delle correlazioni tra le temperature delle stazioni
- studio dell’andamento temporale del carico elettrico

### `models`
Raccoglie i diversi modelli predittivi implementati e confrontati.

#### `polynomial_regression`
Implementazioni di modelli di regressione polinomiale:
- **univariata** (temperatura media come unico regressore)
- **multivariata** (temperatura media e ora del giorno come regressori)
- **stepwise regression** per la selezione automatica dei regressori
  
Viene inoltre affrontato il problema della **ciclicità del tempo** introducendo termini di Fourier (seno e coseno dell’ora del giorno).

#### `mlp`
Modelli non lineari basati su **reti neurali MLP (Multi-Layer Perceptron)**:
- MLP con temperatura media e seno e coseno dell'ora del giorno in input
- MLP con 25 temperature e seno e coseno dell'ora del giorno in input  
L’addestramento è effettuato tramite **Levenberg–Marquardt backpropagation**, con normalizzazione Min-Max e scelta del numero di neuroni tramite **k-fold cross-validation** e regola 1-SD.

---

## Modello finale e predizione

La cartella `predict_function` contiene:
- `modello_finale.mat`: modello MLP finale addestrato
- `predict_load.m`: funzione MATLAB per effettuare la previsione del carico elettrico
- `istruzioni.txt`: istruzioni per l’utilizzo della funzione di predizione

Il **modello finale** è una rete neurale MLP con:
- input: 25 temperature + seno e coseno dell’ora del giorno
- architettura: 1 hidden layer con 19 neuroni

---

## Performance sul Test Set

| Modello                                        | RMSE  | MAPE   | R²     |
|------------------------------------------------|-------|--------|--------|
| Polinomio 5° grado + Fourier                   | 17.93 | 10.14% | 0.8507 |
| Stepwise + Fourier                             | 15.91 | 8.61%  | 0.8828 |
| MLP (8 neuroni, temperatura media)             | 14.93 | 8.06%  | 0.8968 |
| MLP (8 neuroni, 25 temperature)                | 10.79 | 6.03%  | 0.9460 |
| **MLP (19 neuroni, 25 temperature)**           | 10.35 | 5.79%  | 0.9503 |

## Requisiti

- MATLAB con ToolBox:
  - Deep Learning ToolBox
  - Parallel Computing ToolBox

---

## Autore

**Mirko Bocciarelli**   
Università degli Studi di Pavia  
Dipartimento di Ingegneria Industriale e dell’Informazione  

---

## Licenza

Il progetto è distribuito sotto licenza specificata nel file `LICENSE`.
