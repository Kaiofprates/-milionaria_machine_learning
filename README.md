# 🍀 Dashboard +Milionária ML

Um painel interativo em Python com capacidades de Machine Learning para análise e predição da loteria +Milionária.

## 📋 Funcionalidades

### 🎯 Predição de Números
- **5 modelos de ML**: Random Forest, Gradient Boosting, XGBoost, LightGBM e Regressão Linear
- **Predição inteligente**: Baseada em padrões históricos e features estatísticas
- **Interface visual**: Números e trevos exibidos em formato de bolas coloridas
- **Confiança do modelo**: Indicador de confiabilidade da predição

### 📊 Análise de Dados
- **Análise de frequência**: Números e trevos mais/menos sorteados
- **Estatísticas descritivas**: Soma média, amplitude, distribuição par/ímpar
- **Visualizações interativas**: Gráficos de barras, heatmaps e distribuições
- **Correlação entre números**: Análise de padrões entre posições

### 🤖 Treinamento de Modelos
- **Treinamento automático**: Interface para treinar todos os modelos
- **Métricas de performance**: MAE, MSE, RMSE para cada modelo
- **Comparação visual**: Gráficos de performance entre modelos
- **Feature importance**: Importância das variáveis para modelos tree-based

### 📈 Backtesting
- **Teste regressivo**: Avaliação da performance histórica dos modelos
- **Métricas de acerto**: Taxa de sucesso, média de acertos, máximo de acertos
- **Análise detalhada**: Visualização sorteio por sorteio
- **Comparação de modelos**: Performance relativa no backtesting

## 🚀 Como Usar

### 1. Instalação

```bash
# Clone ou baixe o projeto
cd mais_milionaria

# Instale as dependências
pip install -r requirements.txt
```

### 2. Preparação dos Dados

Coloque o arquivo Excel com os dados históricos da +Milionária na pasta do projeto com o nome `+Milionária (2).xlsx`.

**Formato esperado do Excel:**
- Colunas com números: `Num1`, `Num2`, `Num3`, `Num4`, `Num5`, `Num6`
- Colunas com trevos: `Trevo1`, `Trevo2`
- Coluna com data: `Data`
- Coluna com número do concurso: `Concurso`

### 3. Executar o Dashboard

```bash
streamlit run app.py
```

O dashboard será aberto automaticamente no seu navegador em `http://localhost:8501`

## 📁 Estrutura do Projeto

```
mais_milionaria/
├── app.py                          # Dashboard principal Streamlit
├── data_loader.py                  # Carregamento e processamento de dados
├── ml_models.py                    # Modelos de Machine Learning
├── visualizations.py              # Componentes de visualização
├── requirements.txt                # Dependências Python
├── +Milionária (2).xlsx           # Dados históricos (não incluído)
├── +milionaria_funcionamento.md   # Documentação da loteria
└── README.md                      # Este arquivo
```

## 🔧 Componentes Técnicos

### Data Loader (`data_loader.py`)
- **MilionariaDataLoader**: Classe para carregar e processar dados do Excel
- **Features automáticas**: Extração de padrões estatísticos
- **Análise de frequência**: Contagem de ocorrências de números e trevos
- **Dados de exemplo**: Geração automática se o arquivo não for encontrado

### Modelos ML (`ml_models.py`)
- **MilionariaPredictor**: Classe principal para predições
- **5 algoritmos**: Random Forest, Gradient Boosting, XGBoost, LightGBM, Linear Regression
- **Features engenheiradas**: Frequência, padrões temporais, estatísticas
- **Backtesting integrado**: Teste de performance histórica

### Visualizações (`visualizations.py`)
- **MilionariaVisualizer**: Classe para gráficos interativos
- **Plotly integration**: Gráficos responsivos e interativos
- **Múltiplos tipos**: Barras, heatmaps, séries temporais, comparações

## 📊 Features de Machine Learning

### Features Utilizadas
1. **Frequência temporal**: Ocorrência de números em janelas de 5, 10 e 20 sorteios
2. **Padrões estatísticos**: Soma, média, desvio padrão, amplitude
3. **Distribuição**: Números pares/ímpares, distribuição por dezenas
4. **Features temporais**: Mês, dia da semana, dias desde início
5. **Lags**: Valores de sorteios anteriores

### Modelos Implementados
- **Random Forest**: Ensemble robusto, boa para features categóricas
- **Gradient Boosting**: Boosting sequencial, captura padrões complexos
- **XGBoost**: Otimizado, regularização avançada
- **LightGBM**: Eficiente, boa para datasets grandes
- **Linear Regression**: Baseline simples para comparação

## ⚠️ Avisos Importantes

### Sobre Loterias
- **Jogos de azar**: Resultados são fundamentalmente aleatórios
- **Nenhuma garantia**: ML não pode prever números com certeza
- **Entretenimento**: Use apenas para fins educacionais
- **Jogue responsavelmente**: Nunca aposte mais do que pode perder

### Limitações Técnicas
- **Dados limitados**: Performance depende da qualidade dos dados históricos
- **Overfitting**: Modelos podem se ajustar demais aos dados de treino
- **Aleatoriedade**: Loterias são inerentemente imprevisíveis
- **Viés de confirmação**: Sucessos ocasionais não validam o método

## 🛠️ Dependências

- **streamlit**: Interface web interativa
- **pandas**: Manipulação de dados
- **numpy**: Computação numérica
- **scikit-learn**: Algoritmos de ML básicos
- **xgboost**: Gradient boosting otimizado
- **lightgbm**: Gradient boosting eficiente
- **plotly**: Visualizações interativas
- **matplotlib/seaborn**: Gráficos adicionais
- **openpyxl**: Leitura de arquivos Excel

## 📈 Como Interpretar os Resultados

### Métricas de Treinamento
- **MAE (Mean Absolute Error)**: Erro médio absoluto - quanto menor, melhor
- **RMSE (Root Mean Square Error)**: Raiz do erro quadrático médio - penaliza erros grandes

### Métricas de Backtesting
- **Média de Acertos**: Número médio de acertos por sorteio
- **Taxa de Sucesso 3+**: Porcentagem de sorteios com 3 ou mais acertos
- **Máximo de Acertos**: Maior número de acertos em um único sorteio

### Interpretação da Confiança
- **Alta (>70%)**: Modelo teve boa performance no treino
- **Média (50-70%)**: Performance moderada
- **Baixa (<50%)**: Performance limitada

## 🎯 Dicas de Uso

1. **Dados de qualidade**: Use dados históricos completos e corretos
2. **Multiple modelos**: Compare resultados de diferentes algoritmos
3. **Backtesting**: Sempre avalie performance histórica antes de confiar
4. **Combine com intuição**: Use como ferramenta auxiliar, não decisão final
5. **Atualize regularmente**: Retreine modelos com novos dados

## 📞 Suporte

Este é um projeto educacional. Para dúvidas técnicas:
1. Verifique se todas as dependências estão instaladas
2. Confirme o formato dos dados de entrada
3. Execute cada módulo separadamente para identificar erros
4. Consulte a documentação das bibliotecas utilizadas

---

**Desenvolvido para fins educacionais e de entretenimento. Use com responsabilidade!** 🍀
