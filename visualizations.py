import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class MilionariaVisualizer:
    """Classe para visualizações da +Milionária"""
    
    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8'
        }
    
    def plot_number_frequency(self, freq_data: dict, title: str = "Frequência dos Números") -> go.Figure:
        """Gráfico de barras da frequência dos números"""
        if not freq_data:
            return None
        
        df = pd.DataFrame(list(freq_data.items()), columns=['Número', 'Frequência'])
        df = df.sort_values('Número')
        
        fig = px.bar(
            df, 
            x='Número', 
            y='Frequência',
            title=title,
            color='Frequência',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            xaxis_title="Número",
            yaxis_title="Frequência",
            showlegend=False,
            height=400
        )
        
        return fig
    
    def plot_clover_frequency(self, freq_data: dict, title: str = "Frequência dos Trevos") -> go.Figure:
        """Gráfico de barras da frequência dos trevos"""
        if not freq_data:
            return None
        
        df = pd.DataFrame(list(freq_data.items()), columns=['Trevo', 'Frequência'])
        df = df.sort_values('Trevo')
        
        fig = px.bar(
            df, 
            x='Trevo', 
            y='Frequência',
            title=title,
            color_discrete_sequence=[self.colors['success']]
        )
        
        fig.update_layout(
            xaxis_title="Trevo",
            yaxis_title="Frequência",
            showlegend=False,
            height=400
        )
        
        return fig
    
    def plot_sum_distribution(self, data: pd.DataFrame) -> go.Figure:
        """Distribuição da soma dos números"""
        if 'soma_numeros' not in data.columns:
            return None
        
        fig = px.histogram(
            data, 
            x='soma_numeros',
            nbins=30,
            title="Distribuição da Soma dos Números",
            color_discrete_sequence=[self.colors['primary']]
        )
        
        # Adiciona linha da média
        mean_sum = data['soma_numeros'].mean()
        fig.add_vline(
            x=mean_sum, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Média: {mean_sum:.1f}"
        )
        
        fig.update_layout(
            xaxis_title="Soma dos Números",
            yaxis_title="Frequência",
            height=400
        )
        
        return fig
    
    def plot_even_odd_distribution(self, data: pd.DataFrame) -> go.Figure:
        """Distribuição de números pares e ímpares"""
        if 'pares' not in data.columns:
            return None
        
        # Conta a distribuição
        distribution = data['pares'].value_counts().sort_index()
        
        fig = px.bar(
            x=distribution.index,
            y=distribution.values,
            title="Distribuição de Números Pares por Sorteio",
            labels={'x': 'Quantidade de Pares', 'y': 'Frequência'},
            color_discrete_sequence=[self.colors['secondary']]
        )
        
        fig.update_layout(height=400)
        
        return fig
    
    def plot_decade_distribution(self, data: pd.DataFrame) -> go.Figure:
        """Distribuição por dezenas"""
        decade_cols = [col for col in data.columns if col.startswith('dezena_')]
        
        if not decade_cols:
            return None
        
        # Calcula médias por dezena
        decade_means = {}
        for col in decade_cols:
            decade_num = col.split('_')[1]
            decade_means[f"Dezena {decade_num}"] = data[col].mean()
        
        fig = px.bar(
            x=list(decade_means.keys()),
            y=list(decade_means.values()),
            title="Distribuição Média por Dezenas",
            color_discrete_sequence=[self.colors['info']]
        )
        
        fig.update_layout(
            xaxis_title="Dezena",
            yaxis_title="Média de Números por Sorteio",
            height=400
        )
        
        return fig
    
    def plot_correlation_heatmap(self, data: pd.DataFrame) -> go.Figure:
        """Heatmap de correlação entre posições"""
        number_cols = [col for col in data.columns if 'Num' in col and not 'soma' in col.lower()]
        
        if len(number_cols) < 2:
            return None
        
        corr_matrix = data[number_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Correlação entre Posições dos Números",
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def plot_time_series(self, data: pd.DataFrame, column: str, title: str = None) -> go.Figure:
        """Série temporal de uma coluna"""
        if 'Data' not in data.columns or column not in data.columns:
            return None
        
        if title is None:
            title = f"Evolução de {column} ao Longo do Tempo"
        
        fig = px.line(
            data, 
            x='Data', 
            y=column,
            title=title,
            color_discrete_sequence=[self.colors['primary']]
        )
        
        # Adiciona linha de tendência
        fig.add_scatter(
            x=data['Data'],
            y=data[column].rolling(window=10, center=True).mean(),
            mode='lines',
            name='Média Móvel (10)',
            line=dict(color=self.colors['danger'], dash='dash')
        )
        
        fig.update_layout(height=400)
        
        return fig
    
    def plot_model_comparison(self, results: dict) -> go.Figure:
        """Comparação de performance dos modelos"""
        if not results:
            return None
        
        models = list(results.keys())
        mae_values = [results[model].get('mae', 0) for model in models]
        rmse_values = [results[model].get('rmse', 0) for model in models]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Erro Médio Absoluto (MAE)', 'Raiz do Erro Quadrático Médio (RMSE)')
        )
        
        fig.add_trace(
            go.Bar(x=models, y=mae_values, name='MAE', marker_color=self.colors['primary']),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=models, y=rmse_values, name='RMSE', marker_color=self.colors['secondary']),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Comparação de Performance dos Modelos",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def plot_backtest_results(self, backtest_results: dict) -> go.Figure:
        """Visualização dos resultados do backtesting"""
        if not backtest_results:
            return None
        
        models = list(backtest_results.keys())
        success_rates = []
        avg_hits = []
        
        for model in models:
            if 'taxa_sucesso_3+' in backtest_results[model]:
                success_rates.append(backtest_results[model]['taxa_sucesso_3+'] * 100)
            else:
                success_rates.append(0)
            
            if 'media_acertos' in backtest_results[model]:
                avg_hits.append(backtest_results[model]['media_acertos'])
            else:
                avg_hits.append(0)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Taxa de Sucesso 3+ Acertos (%)', 'Média de Acertos por Sorteio')
        )
        
        fig.add_trace(
            go.Bar(x=models, y=success_rates, name='Taxa Sucesso', marker_color=self.colors['success']),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=models, y=avg_hits, name='Média Acertos', marker_color=self.colors['info']),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Resultados do Backtesting",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def plot_prediction_confidence(self, predictions: list) -> go.Figure:
        """Gráfico de confiança das predições"""
        if not predictions:
            return None
        
        df = pd.DataFrame(predictions)
        
        if 'confianca' not in df.columns:
            return None
        
        fig = px.line(
            df, 
            y='confianca',
            title="Evolução da Confiança das Predições",
            color_discrete_sequence=[self.colors['warning']]
        )
        
        fig.update_layout(
            xaxis_title="Predição",
            yaxis_title="Confiança",
            height=400
        )
        
        return fig
    
    def create_number_grid(self, freq_data: dict, max_freq: int = None) -> go.Figure:
        """Grid visual dos números com intensidade baseada na frequência"""
        if not freq_data:
            return None
        
        # Cria grid 10x5 para números 1-50
        grid = np.zeros((5, 10))
        
        if max_freq is None:
            max_freq = max(freq_data.values()) if freq_data else 1
        
        for num, freq in freq_data.items():
            if 1 <= num <= 50:
                row = (num - 1) // 10
                col = (num - 1) % 10
                grid[row, col] = freq / max_freq
        
        # Cria anotações com os números
        annotations = []
        for i in range(5):
            for j in range(10):
                num = i * 10 + j + 1
                annotations.append(
                    dict(
                        x=j, y=i,
                        text=str(num),
                        showarrow=False,
                        font=dict(color="white" if grid[i, j] > 0.5 else "black", size=12)
                    )
                )
        
        fig = go.Figure(data=go.Heatmap(
            z=grid,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Frequência Relativa")
        ))
        
        fig.update_layout(
            title="Grid de Frequência dos Números (1-50)",
            annotations=annotations,
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            height=300,
            width=600
        )
        
        return fig
