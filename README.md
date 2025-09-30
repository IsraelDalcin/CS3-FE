## Funcionalidades

* Criação automática de **labels** com regras heurísticas.
* Extração de **features** textuais e numéricas.
* Treinamento de modelos:

  * Random Forest
  * Logistic Regression
  * Suporte para **GridSearchCV** para otimização de hiperparâmetros
* Avaliação de modelos com:

  * Acurácia, precisão, recall e F1-score
  * Matriz de confusão
  * Curva ROC (para classificação binária)
* Predição de novos produtos.
* Análise de **nível de risco** baseado em predição, confiança do modelo, preço e vendedor.
* Salvamento e carregamento de modelos treinados.
* Logs detalhados de execução em `./content/logs/ai_classifier.log`.
* Exportação de resultados em CSV (`produtos_com_analise_ia.csv`).

---

## Instalação

Requer Python 3.10 ou superior.

Instale as dependências:

```bash
pip install -r requirements.txt
```

## Estrutura do Projeto

```
.
├── content/
│   ├── dataset_hp.csv       # Dataset de produtos
│   └── logs/
│       └── ai_classifier.log
├── resultados/
│   └── produtos_com_analise_ia.csv  # Resultados com predições e nível de risco
├── piracy_detection.py       # Código principal
└── requirements.txt
```

---

## Uso

### Executar o classificador

```bash
python piracy_detection.py
```

O script realiza:

1. Carregamento do dataset (`dataset_hp.csv`).
2. Treinamento do modelo (normal ou com GridSearchCV).
3. Comparação dos melhores modelos.
4. Geração de métricas e gráficos.
5. Predição dos produtos e análise de risco.
6. Exportação dos resultados em CSV.
7. Salvamento do modelo treinado (`modelo_deteccao_pirataria.pkl`).

---

### Alternar GridSearch

No arquivo `piracy_detection.py`:

```python
usar_gridsearch = True  # Altere para False para treinamento normal
```

---

### Predição de novos produtos

```python
classifier = PiracyDetectionClassifier()
classifier.load_model("resultados/modelo_deteccao_pirataria.pkl")
df_new = pd.read_csv("novos_produtos.csv")
df_pred = classifier.prever(df_new)
df_pred = classifier.analyze_risk_level(df_pred)
```

---

## Logs

Logs detalhados são gravados em:

```
./content/logs/ai_classifier.log
```

Contendo informações sobre:

* Criação de features
* Treinamento de modelos
* Resultados de GridSearch
* Acurácia e métricas

---

## Métricas

O classificador gera:

* **Accuracy (Acurácia)**
* **Precision (Precisão)**
* **Recall**
* **F1-score**
* **Matriz de Confusão**
* **Curva ROC** (quando aplicável)

Além disso, o **nível de risco** é classificado como:

* BAIXO
* MÉDIO
* ALTO

Baseado em pontuação combinando predição, confiança, preço e vendedor.
