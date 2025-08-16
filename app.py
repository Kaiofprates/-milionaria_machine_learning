import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import MilionariaDataLoader
from ml_models import MilionariaPredictor
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Dashboard +Milionária ML",
    page_icon="🍀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .number-ball {
        display: inline-block;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background-color: #1f77b4;
        color: white;
        text-align: center;
        line-height: 50px;
        margin: 5px;
        font-weight: bold;
        font-size: 18px;
    }
    .clover-ball {
        display: inline-block;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background-color: #2ca02c;
        color: white;
        text-align: center;
        line-height: 50px;
        margin: 5px;
        font-weight: bold;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(uploaded_file=None):
    """Carrega os dados com cache"""
    if uploaded_file is not None:
        # Usa arquivo carregado pelo usuário
        loader = MilionariaDataLoader(uploaded_file)
        data = loader.load_data()
        processed_data = loader.preprocess_data()
        return loader, data, processed_data
    else:
        # Tenta carregar arquivo local
        file_path = "+Milionária (2).xlsx"
        if not os.path.exists(file_path):
            st.warning("Arquivo Excel não encontrado. Usando dados de exemplo.")
        
        loader = MilionariaDataLoader(file_path)
        data = loader.load_data()
        processed_data = loader.preprocess_data()
        return loader, data, processed_data

def display_prediction(prediction):
    """Exibe a predição de forma visual"""
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    st.markdown("### 🎯 Predição do Próximo Sorteio")
    
    # Números principais
    st.markdown("**Números Principais:**")
    numbers_html = ""
    for num in prediction['numeros']:
        numbers_html += f'<span class="number-ball">{num:02d}</span>'
    st.markdown(numbers_html, unsafe_allow_html=True)
    
    # Trevos
    st.markdown("**Trevos:**")
    clovers_html = ""
    for trevo in prediction['trevos']:
        clovers_html += f'<span class="clover-ball">{trevo}</span>'
    st.markdown(clovers_html, unsafe_allow_html=True)
    
    st.markdown(f"**Modelo:** {prediction['modelo_usado']}")
    st.markdown(f"**Confiança:** {prediction['confianca']:.1%}")
    st.markdown('</div>', unsafe_allow_html=True)

def create_frequency_chart(freq_data, title, color):
    """Cria gráfico de frequência"""
    if not freq_data:
        return None
    
    df = pd.DataFrame(list(freq_data.items()), columns=['Número', 'Frequência'])
    fig = px.bar(df, x='Número', y='Frequência', title=title, color_discrete_sequence=[color])
    fig.update_layout(height=400)
    return fig

def create_heatmap(data):
    """Cria heatmap de correlação dos números"""
    number_cols = [col for col in data.columns if 'Num' in col]
    if len(number_cols) < 2:
        return None
    
    corr_matrix = data[number_cols].corr()
    fig = px.imshow(corr_matrix, title="Correlação entre Posições dos Números", 
                    color_continuous_scale='RdBu_r')
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🍀 Dashboard +Milionária ML</h1>', unsafe_allow_html=True)
    
    # Data source info
    st.info("""
    📥 **Para usar dados reais:** Baixe o arquivo Excel atualizado da Caixa Econômica Federal e coloque na pasta do projeto com o nome `+Milionária (2).xlsx`
    
    🔗 **Link oficial:** [Baixar dados históricos da +Milionária](https://loterias.caixa.gov.br/Paginas/Mais-Milionaria.aspx)
    
    ⚠️ **Importante:** O arquivo é atualizado automaticamente após cada sorteio. Baixe sempre a versão mais recente para análises precisas.
    """)
    
    # Sidebar
    st.sidebar.title("⚙️ Configurações")
    
    # Upload de arquivo
    st.sidebar.subheader("📁 Carregar Dados")
    uploaded_file = st.sidebar.file_uploader(
        "Faça upload do arquivo Excel da +Milionária:",
        type=['xlsx'],
        help="Baixe o arquivo oficial em: loterias.caixa.gov.br/Paginas/Mais-Milionaria.aspx"
    )
    
    if uploaded_file is not None:
        st.sidebar.success("✅ Arquivo carregado com sucesso!")
        st.sidebar.info(f"📄 Arquivo: {uploaded_file.name}")
    
    # Carrega dados
    try:
        loader, data, processed_data = load_data(uploaded_file)
        data_info = loader.get_data_info()
        
        if uploaded_file is not None:
            st.sidebar.success(f"✅ {data_info['total_sorteios']} sorteios do arquivo carregado")
        else:
            st.sidebar.success(f"✅ {data_info['total_sorteios']} sorteios carregados")
        
        if data_info['periodo']:
            st.sidebar.info(f"📅 Período: {data_info['periodo']['inicio']} a {data_info['periodo']['fim']}")
        
    except Exception as e:
        if uploaded_file is None:
            st.sidebar.error("❌ Arquivo de dados não encontrado")
            st.sidebar.markdown("""
            **Para usar dados reais:**
            1. Acesse: [Site da Caixa](https://loterias.caixa.gov.br/Paginas/Mais-Milionaria.aspx)
            2. Baixe o arquivo Excel
            3. Use o botão acima para carregar
            
            **Ou:**
            - Renomeie para: `+Milionária (2).xlsx`
            - Coloque na pasta do projeto
            """)
            st.sidebar.warning("⚠️ Usando dados de exemplo")
        else:
            st.sidebar.error("❌ Erro ao processar arquivo carregado")
            st.sidebar.info("Verifique se é o arquivo correto da Caixa")
        return
    
    # Menu principal
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Predição", "📊 Análise de Dados", "🤖 Treinamento", "📈 Backtesting", "ℹ️ Informações"
    ])
    
    # Tab 1: Predição
    with tab1:
        st.header("Predição do Próximo Sorteio")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Configurações")
            
            model_options = {
                'Random Forest': 'random_forest',
                'Gradient Boosting': 'gradient_boosting',
                'XGBoost': 'xgboost',
                'LightGBM': 'lightgbm',
                'Regressão Linear': 'linear_regression'
            }
            
            selected_model = st.selectbox(
                "Escolha o modelo:",
                options=list(model_options.keys()),
                index=0
            )
            
            if st.button("🎲 Gerar Predição", type="primary"):
                with st.spinner("Treinando modelo e gerando predição..."):
                    try:
                        predictor = MilionariaPredictor()
                        predictor.train_models(processed_data)
                        
                        prediction = predictor.predict_next_draw(
                            processed_data, 
                            model_options[selected_model]
                        )
                        
                        st.session_state['prediction'] = prediction
                        st.success("Predição gerada com sucesso!")
                        
                    except Exception as e:
                        st.error(f"Erro ao gerar predição: {e}")
        
        with col1:
            if 'prediction' in st.session_state:
                display_prediction(st.session_state['prediction'])
            else:
                st.info("Clique em 'Gerar Predição' para ver o resultado")
        
        # Últimos sorteios
        st.subheader("📋 Últimos Sorteios")
        last_draws = loader.get_last_draws(10)
        
        if not last_draws.empty:
            # Formatar para exibição
            display_cols = []
            for col in last_draws.columns:
                if any(x in col.lower() for x in ['concurso', 'data', 'num', 'trevo']):
                    display_cols.append(col)
            
            if display_cols:
                st.dataframe(last_draws[display_cols].tail(10), use_container_width=True)
    
    # Tab 2: Análise de Dados
    with tab2:
        st.header("Análise dos Dados Históricos")
        
        # Análise de frequência
        freq_analysis = loader.get_frequency_analysis()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'numeros' in freq_analysis:
                fig_nums = create_frequency_chart(
                    freq_analysis['numeros'], 
                    "Frequência dos Números Principais", 
                    "#1f77b4"
                )
                if fig_nums:
                    st.plotly_chart(fig_nums, use_container_width=True)
        
        with col2:
            if 'trevos' in freq_analysis:
                fig_trevos = create_frequency_chart(
                    freq_analysis['trevos'], 
                    "Frequência dos Trevos", 
                    "#2ca02c"
                )
                if fig_trevos:
                    st.plotly_chart(fig_trevos, use_container_width=True)
        
        # Estatísticas gerais
        st.subheader("📊 Estatísticas Gerais")
        
        if not processed_data.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'soma_numeros' in processed_data.columns:
                    avg_sum = processed_data['soma_numeros'].mean()
                    st.metric("Soma Média", f"{avg_sum:.1f}")
            
            with col2:
                if 'pares' in processed_data.columns:
                    avg_pares = processed_data['pares'].mean()
                    st.metric("Média de Pares", f"{avg_pares:.1f}")
            
            with col3:
                if 'amplitude' in processed_data.columns:
                    avg_amplitude = processed_data['amplitude'].mean()
                    st.metric("Amplitude Média", f"{avg_amplitude:.1f}")
            
            with col4:
                if 'soma_trevos' in processed_data.columns:
                    avg_trevos = processed_data['soma_trevos'].mean()
                    st.metric("Soma Média Trevos", f"{avg_trevos:.1f}")
        
        # Heatmap de correlação
        st.subheader("🔥 Correlação entre Números")
        heatmap_fig = create_heatmap(data)
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # Tab 3: Treinamento
    with tab3:
        st.header("Treinamento dos Modelos")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Configurações de Treinamento")
            
            train_all = st.checkbox("Treinar todos os modelos", value=True)
            
            if not train_all:
                selected_models = st.multiselect(
                    "Selecione os modelos:",
                    options=list(model_options.keys()),
                    default=list(model_options.keys())[:2]
                )
            
            if st.button("🚀 Iniciar Treinamento", type="primary"):
                with st.spinner("Treinando modelos..."):
                    try:
                        predictor = MilionariaPredictor()
                        results = predictor.train_models(processed_data)
                        
                        st.session_state['training_results'] = results
                        st.session_state['predictor'] = predictor
                        st.success("Treinamento concluído!")
                        
                    except Exception as e:
                        st.error(f"Erro no treinamento: {e}")
        
        with col2:
            if 'training_results' in st.session_state:
                st.subheader("📈 Resultados do Treinamento")
                
                results_df = pd.DataFrame(st.session_state['training_results']).T
                results_df = results_df.round(4)
                
                st.dataframe(results_df, use_container_width=True)
                
                # Gráfico de comparação
                if not results_df.empty:
                    fig = px.bar(
                        results_df.reset_index(), 
                        x='index', 
                        y='mae',
                        title="Erro Médio Absoluto por Modelo",
                        labels={'index': 'Modelo', 'mae': 'MAE'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Backtesting
    with tab4:
        st.header("📈 Teste Regressivo dos Modelos")
        
        st.markdown("""
        O **backtesting** avalia como os modelos teriam performado em sorteios passados.
        É a melhor forma de validar a eficácia dos algoritmos de predição.
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Configurações")
            
            test_size = st.slider(
                "Sorteios para teste:",
                min_value=5,
                max_value=min(50, len(data) // 2),
                value=20,
                help="Número de sorteios mais recentes para testar os modelos"
            )
            
            st.info(f"📊 **Dados disponíveis:** {len(data)} sorteios\n📋 **Para treino:** {len(data) - test_size} sorteios\n🎯 **Para teste:** {test_size} sorteios")
            
            if st.button("🔍 Executar Backtesting", type="primary", use_container_width=True):
                with st.spinner("Executando backtesting... Isso pode levar alguns minutos."):
                    try:
                        predictor = MilionariaPredictor()
                        backtest_results = predictor.backtest(processed_data, test_size)
                        
                        st.session_state['backtest_results'] = backtest_results
                        st.success("✅ Backtesting concluído com sucesso!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Erro no backtesting: {str(e)}")
                        st.info("💡 Dica: Verifique se há dados suficientes para o teste.")
        
        with col2:
            if 'backtest_results' in st.session_state:
                st.subheader("📊 Resultados do Backtesting")
                
                results = st.session_state['backtest_results']
                
                # Métricas principais em destaque
                if results:
                    best_model = None
                    best_score = 0
                    
                    for model_name, model_results in results.items():
                        if 'taxa_sucesso_3+' in model_results and model_results['taxa_sucesso_3+'] > best_score:
                            best_score = model_results['taxa_sucesso_3+']
                            best_model = model_name
                    
                    if best_model:
                        st.success(f"🏆 **Melhor Modelo:** {best_model} ({best_score:.1%} de taxa de sucesso 3+)")
                
                # Tabela resumo
                summary_data = []
                for model_name, model_results in results.items():
                    if 'media_acertos' in model_results:
                        summary_data.append({
                            'Modelo': model_name,
                            'Média de Acertos': f"{model_results['media_acertos']:.2f}",
                            'Máximo de Acertos': model_results['max_acertos'],
                            'Sorteios com 3+ Acertos': model_results['acertos_3_ou_mais'],
                            'Taxa de Sucesso 3+': f"{model_results['taxa_sucesso_3+']:.1%}",
                            'Total de Acertos': model_results['acertos_totais']
                        })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Gráficos de desempenho
                    col_chart1, col_chart2 = st.columns(2)
                    
                    with col_chart1:
                        fig1 = px.bar(
                            summary_df, 
                            x='Modelo', 
                            y='Taxa de Sucesso 3+',
                            title="Taxa de Sucesso (3+ Acertos)",
                            color='Taxa de Sucesso 3+',
                            color_continuous_scale='Viridis'
                        )
                        fig1.update_layout(height=400)
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col_chart2:
                        fig2 = px.bar(
                            summary_df, 
                            x='Modelo', 
                            y='Média de Acertos',
                            title="Média de Acertos por Sorteio",
                            color='Média de Acertos',
                            color_continuous_scale='Blues'
                        )
                        fig2.update_layout(height=400)
                        st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("👈 Execute o backtesting para ver os resultados aqui")
        
        # Seção expandida com detalhes
        if 'backtest_results' in st.session_state:
            st.markdown("---")
            
            with st.expander("🔍 Análise Detalhada por Modelo", expanded=False):
                results = st.session_state['backtest_results']
                
                selected_model_detail = st.selectbox(
                    "Selecione um modelo para análise detalhada:",
                    options=list(results.keys()),
                    key="detailed_model_select"
                )
                
                if selected_model_detail in results:
                    model_detail = results[selected_model_detail]
                    
                    # Métricas do modelo
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Média de Acertos", f"{model_detail.get('media_acertos', 0):.2f}")
                    with col2:
                        st.metric("Máximo de Acertos", model_detail.get('max_acertos', 0))
                    with col3:
                        st.metric("Sorteios 3+ Acertos", model_detail.get('acertos_3_ou_mais', 0))
                    with col4:
                        st.metric("Taxa Sucesso 3+", f"{model_detail.get('taxa_sucesso_3+', 0):.1%}")
                    
                    # Gráfico de evolução dos acertos
                    if 'acertos_por_sorteio' in model_detail:
                        acertos_data = pd.DataFrame({
                            'Sorteio': range(1, len(model_detail['acertos_por_sorteio']) + 1),
                            'Acertos': model_detail['acertos_por_sorteio']
                        })
                        
                        fig_evolution = px.line(
                            acertos_data, 
                            x='Sorteio', 
                            y='Acertos',
                            title=f"Evolução dos Acertos - {selected_model_detail}",
                            markers=True
                        )
                        fig_evolution.add_hline(y=3, line_dash="dash", line_color="red", 
                                              annotation_text="Limite de Sucesso (3 acertos)")
                        st.plotly_chart(fig_evolution, use_container_width=True)
                    
                    # Tabela detalhada das predições
                    if 'predicoes' in model_detail:
                        st.subheader("📋 Predições Detalhadas")
                        predicoes_df = pd.DataFrame(model_detail['predicoes'])
                        
                        # Destaca linhas com 3+ acertos
                        def highlight_success(row):
                            if row['acertos'] >= 3:
                                return ['background-color: #d4edda'] * len(row)
                            elif row['acertos'] >= 2:
                                return ['background-color: #fff3cd'] * len(row)
                            else:
                                return [''] * len(row)
                        
                        styled_df = predicoes_df.style.apply(highlight_success, axis=1)
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Distribuição de acertos
                        acertos_dist = predicoes_df['acertos'].value_counts().sort_index()
                        fig_dist = px.bar(
                            x=acertos_dist.index, 
                            y=acertos_dist.values,
                            title="Distribuição de Acertos",
                            labels={'x': 'Número de Acertos', 'y': 'Frequência'}
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Tab 5: Informações
    with tab5:
        st.header("ℹ️ Informações sobre a +Milionária")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎮 Como Funciona")
            st.markdown("""
            A **+Milionária** é uma loteria com dupla matriz:
            - **6 números** de 1 a 50
            - **2 trevos** de 1 a 6
            
            **Probabilidades (aposta simples):**
            - Prêmio máximo: 1 em 238.360.500
            - 4 acertos + 2 trevos: 1 em 16.798
            - 2 acertos + 1 trevo: 1 em 15
            """)
            
            st.subheader("💰 Faixas de Premiação")
            st.markdown("""
            **10 faixas de premiação:**
            1. 6 + 2 trevos (62% - variável)
            2. 6 + 1 ou 0 trevos (10% - variável)
            3. 5 + 2 trevos (8% - variável)
            4. 5 + 1 ou 0 trevos (8% - variável)
            5. 4 + 2 trevos (6% - variável)
            6. 4 + 1 ou 0 trevos (6% - variável)
            7. 3 + 2 trevos (R$ 50,00 - fixo)
            8. 3 + 1 trevo (R$ 24,00 - fixo)
            9. 2 + 2 trevos (R$ 12,00 - fixo)
            10. 2 + 1 trevo (R$ 6,00 - fixo)
            """)
        
        with col2:
            st.subheader("🤖 Sobre os Modelos ML")
            st.markdown("""
            **Modelos Implementados:**
            - **Random Forest**: Ensemble de árvores de decisão
            - **Gradient Boosting**: Boosting sequencial
            - **XGBoost**: Gradient boosting otimizado
            - **LightGBM**: Gradient boosting eficiente
            - **Regressão Linear**: Modelo linear simples
            
            **Features Utilizadas:**
            - Frequência dos números em janelas temporais
            - Padrões estatísticos (soma, média, amplitude)
            - Distribuição por dezenas
            - Números pares/ímpares
            - Features temporais
            """)
            
            st.subheader("📥 Fonte dos Dados")
            st.markdown("""
            **Dados Oficiais da Caixa Econômica Federal:**
            
            🔗 **Link direto:** [loterias.caixa.gov.br/Paginas/Mais-Milionaria.aspx](https://loterias.caixa.gov.br/Paginas/Mais-Milionaria.aspx)
            
            **Como obter os dados:**
            1. Acesse o site oficial da Caixa
            2. Procure por "Resultados" ou "Download"
            3. Baixe o arquivo Excel com histórico completo
            4. Renomeie para `+Milionária (2).xlsx`
            5. Coloque na pasta raiz do projeto
            
            **Atualização:** O arquivo é atualizado automaticamente após cada sorteio (sábados às 20h).
            """)
            
            st.subheader("⚠️ Aviso Importante")
            st.warning("""
            **Este sistema é apenas para fins educacionais e de entretenimento.**
            
            Loterias são jogos de azar e os resultados são aleatórios. 
            Nenhum modelo de machine learning pode prever com certeza 
            os números sorteados.
            
            Jogue com responsabilidade!
            """)

if __name__ == "__main__":
    main()
