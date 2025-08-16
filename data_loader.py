import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import os

class MilionariaDataLoader:
    """Carregador de dados históricos da +Milionária"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.processed_data = None
        
    def load_data(self) -> pd.DataFrame:
        """Carrega os dados do arquivo Excel"""
        try:
            # Verifica se é um arquivo carregado (UploadedFile) ou caminho
            if hasattr(self.file_path, 'read'):
                # É um arquivo carregado pelo Streamlit
                self.data = pd.read_excel(self.file_path)
            elif isinstance(self.file_path, str) and self.file_path.endswith('.xlsx'):
                # É um caminho de arquivo local
                self.data = pd.read_excel(self.file_path)
            else:
                raise ValueError("Formato de arquivo não suportado")
                
            print(f"Dados carregados: {len(self.data)} registros")
            return self.data
            
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            # Cria dados de exemplo se não conseguir carregar
            return self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Cria dados de exemplo para demonstração"""
        np.random.seed(42)
        n_draws = 100
        
        data = []
        for i in range(1, n_draws + 1):
            # Números principais (6 de 50)
            numbers = sorted(np.random.choice(range(1, 51), 6, replace=False))
            # Trevos (2 de 6)
            clovers = sorted(np.random.choice(range(1, 7), 2, replace=False))
            
            data.append({
                'Concurso': i,
                'Data': pd.date_range('2022-05-28', periods=n_draws, freq='W-SAT')[i-1],
                'Num1': numbers[0], 'Num2': numbers[1], 'Num3': numbers[2],
                'Num4': numbers[3], 'Num5': numbers[4], 'Num6': numbers[5],
                'Trevo1': clovers[0], 'Trevo2': clovers[1]
            })
        
        self.data = pd.DataFrame(data)
        print("Usando dados de exemplo (100 sorteios simulados)")
        return self.data
    
    def preprocess_data(self) -> pd.DataFrame:
        """Preprocessa os dados para machine learning"""
        if self.data is None:
            self.load_data()
        
        df = self.data.copy()
        
        # Extrai features dos números
        number_cols = [col for col in df.columns if 'Num' in col or col.startswith('Num')]
        clover_cols = [col for col in df.columns if 'Trevo' in col or 'Clover' in col]
        
        # Se não encontrar colunas específicas, tenta identificar automaticamente
        if not number_cols:
            # Assume que as primeiras 6 colunas numéricas são os números
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Concurso' in numeric_cols:
                numeric_cols.remove('Concurso')
            number_cols = numeric_cols[:6] if len(numeric_cols) >= 6 else []
            clover_cols = numeric_cols[6:8] if len(numeric_cols) >= 8 else []
        
        # Cria features estatísticas
        if number_cols:
            df['soma_numeros'] = df[number_cols].sum(axis=1)
            df['media_numeros'] = df[number_cols].mean(axis=1)
            df['std_numeros'] = df[number_cols].std(axis=1)
            df['min_numero'] = df[number_cols].min(axis=1)
            df['max_numero'] = df[number_cols].max(axis=1)
            df['amplitude'] = df['max_numero'] - df['min_numero']
        
        if clover_cols:
            df['soma_trevos'] = df[clover_cols].sum(axis=1)
        
        # Features de padrões
        if number_cols:
            # Números pares/ímpares
            df['pares'] = df[number_cols].apply(lambda x: sum(x % 2 == 0), axis=1)
            df['impares'] = 6 - df['pares']
            
            # Distribuição por dezenas
            df['dezena_1'] = df[number_cols].apply(lambda x: sum((x >= 1) & (x <= 10)), axis=1)
            df['dezena_2'] = df[number_cols].apply(lambda x: sum((x >= 11) & (x <= 20)), axis=1)
            df['dezena_3'] = df[number_cols].apply(lambda x: sum((x >= 21) & (x <= 30)), axis=1)
            df['dezena_4'] = df[number_cols].apply(lambda x: sum((x >= 31) & (x <= 40)), axis=1)
            df['dezena_5'] = df[number_cols].apply(lambda x: sum((x >= 41) & (x <= 50)), axis=1)
        
        # Features temporais se houver coluna de data
        if 'Data' in df.columns:
            df['Data'] = pd.to_datetime(df['Data'])
            df['ano'] = df['Data'].dt.year
            df['mes'] = df['Data'].dt.month
            df['dia_semana'] = df['Data'].dt.dayofweek
        
        self.processed_data = df
        return df
    
    def get_frequency_analysis(self) -> dict:
        """Análise de frequência dos números"""
        if self.data is None:
            self.load_data()
        
        # Identifica colunas de números
        number_cols = [col for col in self.data.columns if 'Num' in col or col.startswith('Num')]
        clover_cols = [col for col in self.data.columns if 'Trevo' in col or 'Clover' in col]
        
        if not number_cols:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if 'Concurso' in numeric_cols:
                numeric_cols.remove('Concurso')
            number_cols = numeric_cols[:6] if len(numeric_cols) >= 6 else []
            clover_cols = numeric_cols[6:8] if len(numeric_cols) >= 8 else []
        
        analysis = {}
        
        if number_cols:
            # Frequência dos números principais
            all_numbers = []
            for col in number_cols:
                all_numbers.extend(self.data[col].tolist())
            
            freq_numbers = pd.Series(all_numbers).value_counts().sort_index()
            analysis['numeros'] = freq_numbers.to_dict()
        
        if clover_cols:
            # Frequência dos trevos
            all_clovers = []
            for col in clover_cols:
                all_clovers.extend(self.data[col].tolist())
            
            freq_clovers = pd.Series(all_clovers).value_counts().sort_index()
            analysis['trevos'] = freq_clovers.to_dict()
        
        return analysis
    
    def get_last_draws(self, n: int = 10) -> pd.DataFrame:
        """Retorna os últimos n sorteios"""
        if self.data is None:
            self.load_data()
        
        return self.data.tail(n)
    
    def get_data_info(self) -> dict:
        """Retorna informações sobre os dados"""
        if self.data is None:
            self.load_data()
        
        info = {
            'total_sorteios': len(self.data),
            'colunas': list(self.data.columns),
            'periodo': None
        }
        
        if 'Data' in self.data.columns:
            dates = pd.to_datetime(self.data['Data'])
            info['periodo'] = {
                'inicio': dates.min().strftime('%Y-%m-%d'),
                'fim': dates.max().strftime('%Y-%m-%d')
            }
        
        return info
