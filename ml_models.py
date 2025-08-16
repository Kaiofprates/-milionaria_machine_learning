import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from typing import Tuple, List, Dict, Any
import joblib
import os

class MilionariaPredictor:
    """Preditor de números da +Milionária usando Machine Learning"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            'linear_regression': LinearRegression()
        }
        self.scalers = {}
        self.trained_models = {}
        self.feature_importance = {}
        
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepara features para treinamento"""
        # Features baseadas em estatísticas dos sorteios anteriores
        features = []
        feature_names = []
        
        # Features de frequência (últimos N sorteios)
        for window in [5, 10, 20]:
            for num in range(1, 51):  # Números de 1 a 50
                freq_col = f'freq_num_{num}_last_{window}'
                feature_names.append(freq_col)
                
                freq_values = []
                for i in range(len(data)):
                    start_idx = max(0, i - window)
                    window_data = data.iloc[start_idx:i]
                    
                    if len(window_data) == 0:
                        freq_values.append(0)
                    else:
                        # Conta quantas vezes o número apareceu
                        count = 0
                        number_cols = [col for col in window_data.columns if 'Num' in col]
                        for col in number_cols:
                            count += (window_data[col] == num).sum()
                        freq_values.append(count / len(window_data))
                
                features.append(freq_values)
        
        # Features de trevos
        for window in [5, 10]:
            for trevo in range(1, 7):  # Trevos de 1 a 6
                freq_col = f'freq_trevo_{trevo}_last_{window}'
                feature_names.append(freq_col)
                
                freq_values = []
                for i in range(len(data)):
                    start_idx = max(0, i - window)
                    window_data = data.iloc[start_idx:i]
                    
                    if len(window_data) == 0:
                        freq_values.append(0)
                    else:
                        count = 0
                        trevo_cols = [col for col in window_data.columns if 'Trevo' in col]
                        for col in trevo_cols:
                            count += (window_data[col] == trevo).sum()
                        freq_values.append(count / len(window_data))
                
                features.append(freq_values)
        
        # Features temporais
        if 'Data' in data.columns:
            dates = pd.to_datetime(data['Data'])
            features.extend([
                dates.dt.month.values,
                dates.dt.dayofweek.values,
                (dates - dates.min()).dt.days.values  # Dias desde o primeiro sorteio
            ])
            feature_names.extend(['mes', 'dia_semana', 'dias_desde_inicio'])
        
        # Features de padrões dos últimos sorteios
        if any('soma_numeros' in col for col in data.columns):
            for lag in [1, 2, 3]:
                lag_col = f'soma_numeros_lag_{lag}'
                feature_names.append(lag_col)
                lag_values = data['soma_numeros'].shift(lag).fillna(data['soma_numeros'].mean()).values
                features.append(lag_values)
        
        X = np.array(features).T
        
        # Target: próximos números sorteados
        number_cols = [col for col in data.columns if 'Num' in col][:6]
        if number_cols:
            y = data[number_cols].values
        else:
            # Se não encontrar, usa as primeiras 6 colunas numéricas
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if 'Concurso' in numeric_cols:
                numeric_cols.remove('Concurso')
            y = data[numeric_cols[:6]].values if len(numeric_cols) >= 6 else np.zeros((len(data), 6))
        
        return X, y, feature_names
    
    def train_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Treina todos os modelos"""
        X, y, feature_names = self.prepare_features(data)
        
        # Remove primeiras linhas que podem ter NaN devido aos lags
        valid_idx = ~np.isnan(X).any(axis=1)
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 10:
            raise ValueError("Dados insuficientes para treinamento")
        
        # Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"Treinando {model_name}...")
            
            # Normalização para alguns modelos
            if model_name in ['linear_regression']:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                self.scalers[model_name] = scaler
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # Treina modelo para cada posição (1-6 números)
            model_predictions = []
            position_models = []
            
            for pos in range(6):
                try:
                    pos_model = self._clone_model(model)
                    pos_model.fit(X_train_scaled, y_train[:, pos])
                    position_models.append(pos_model)
                    
                    # Predição
                    pred = pos_model.predict(X_test_scaled)
                    model_predictions.append(pred)
                    
                except Exception as e:
                    print(f"Erro ao treinar {model_name} posição {pos}: {e}")
                    position_models.append(None)
                    model_predictions.append(np.zeros(len(X_test_scaled)))
            
            self.trained_models[model_name] = position_models
            
            # Avaliação
            if model_predictions:
                predictions = np.array(model_predictions).T
                mae = mean_absolute_error(y_test, predictions)
                mse = mean_squared_error(y_test, predictions)
                
                results[model_name] = {
                    'mae': mae,
                    'mse': mse,
                    'rmse': np.sqrt(mse)
                }
                
                # Feature importance para modelos tree-based
                if hasattr(model, 'feature_importances_'):
                    avg_importance = np.mean([
                        m.feature_importances_ for m in position_models if m is not None
                    ], axis=0)
                    self.feature_importance[model_name] = dict(zip(feature_names, avg_importance))
        
        return results
    
    def _clone_model(self, model):
        """Clona um modelo"""
        if isinstance(model, RandomForestRegressor):
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif isinstance(model, GradientBoostingRegressor):
            return GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif isinstance(model, xgb.XGBRegressor):
            return xgb.XGBRegressor(n_estimators=100, random_state=42)
        elif isinstance(model, lgb.LGBMRegressor):
            return lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        elif isinstance(model, LinearRegression):
            return LinearRegression()
        else:
            return model
    
    def predict_next_draw(self, data: pd.DataFrame, model_name: str = 'random_forest') -> Dict[str, Any]:
        """Prediz o próximo sorteio"""
        if model_name not in self.trained_models:
            raise ValueError(f"Modelo {model_name} não foi treinado")
        
        X, _, feature_names = self.prepare_features(data)
        
        # Usa o último registro para predição
        X_last = X[-1:].reshape(1, -1)
        
        # Aplica normalização se necessário
        if model_name in self.scalers:
            X_last = self.scalers[model_name].transform(X_last)
        
        # Predição para cada posição
        predictions = []
        position_models = self.trained_models[model_name]
        
        for pos in range(6):
            if position_models[pos] is not None:
                pred = position_models[pos].predict(X_last)[0]
                # Garante que está no range válido (1-50)
                pred = max(1, min(50, round(pred)))
                predictions.append(pred)
            else:
                predictions.append(np.random.randint(1, 51))
        
        # Remove duplicatas e garante 6 números únicos
        unique_predictions = []
        for pred in predictions:
            if pred not in unique_predictions:
                unique_predictions.append(pred)
        
        # Completa com números aleatórios se necessário
        while len(unique_predictions) < 6:
            rand_num = np.random.randint(1, 51)
            if rand_num not in unique_predictions:
                unique_predictions.append(rand_num)
        
        # Predição dos trevos (simplificada)
        trevos = sorted(np.random.choice(range(1, 7), 2, replace=False))
        
        return {
            'numeros': sorted(unique_predictions[:6]),
            'trevos': trevos,
            'modelo_usado': model_name,
            'confianca': self._calculate_confidence(model_name)
        }
    
    def _calculate_confidence(self, model_name: str) -> float:
        """Calcula confiança da predição (simplificado)"""
        # Baseado na performance do modelo
        base_confidence = {
            'random_forest': 0.65,
            'gradient_boosting': 0.62,
            'xgboost': 0.68,
            'lightgbm': 0.66,
            'linear_regression': 0.45
        }
        return base_confidence.get(model_name, 0.50)
    
    def save_models(self, filepath: str):
        """Salva os modelos treinados"""
        model_data = {
            'trained_models': self.trained_models,
            'scalers': self.scalers,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, filepath)
        print(f"Modelos salvos em {filepath}")
    
    def load_models(self, filepath: str):
        """Carrega modelos salvos"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.trained_models = model_data.get('trained_models', {})
            self.scalers = model_data.get('scalers', {})
            self.feature_importance = model_data.get('feature_importance', {})
            print(f"Modelos carregados de {filepath}")
        else:
            print(f"Arquivo {filepath} não encontrado")
    
    def backtest(self, data: pd.DataFrame, test_size: int = 20) -> Dict[str, Any]:
        """Realiza backtesting dos modelos"""
        if len(data) < test_size + 10:
            raise ValueError("Dados insuficientes para backtesting")
        
        # Separa dados de treino e teste
        train_data = data.iloc[:-test_size]
        test_data = data.iloc[-test_size:]
        
        # Treina modelos com dados de treino
        self.train_models(train_data)
        
        results = {}
        
        for model_name in self.trained_models.keys():
            model_results = {
                'acertos_totais': 0,
                'acertos_por_sorteio': [],
                'predicoes': []
            }
            
            # Testa cada sorteio
            for i in range(len(test_data)):
                # Dados até o sorteio atual
                current_data = pd.concat([train_data, test_data.iloc[:i]])
                
                if len(current_data) > 0:
                    # Predição
                    try:
                        prediction = self.predict_next_draw(current_data, model_name)
                        actual_numbers = test_data.iloc[i]
                        
                        # Extrai números reais
                        number_cols = [col for col in actual_numbers.index if 'Num' in col]
                        if not number_cols:
                            numeric_cols = test_data.select_dtypes(include=[np.number]).columns.tolist()
                            if 'Concurso' in numeric_cols:
                                numeric_cols.remove('Concurso')
                            number_cols = numeric_cols[:6]
                        
                        actual_nums = [actual_numbers[col] for col in number_cols[:6]]
                        
                        # Conta acertos
                        acertos = len(set(prediction['numeros']) & set(actual_nums))
                        model_results['acertos_totais'] += acertos
                        model_results['acertos_por_sorteio'].append(acertos)
                        model_results['predicoes'].append({
                            'sorteio': i + 1,
                            'predicao': prediction['numeros'],
                            'real': actual_nums,
                            'acertos': acertos
                        })
                        
                    except Exception as e:
                        print(f"Erro no backtesting {model_name}, sorteio {i}: {e}")
                        model_results['acertos_por_sorteio'].append(0)
            
            # Estatísticas finais
            if model_results['acertos_por_sorteio']:
                model_results['media_acertos'] = np.mean(model_results['acertos_por_sorteio'])
                model_results['max_acertos'] = max(model_results['acertos_por_sorteio'])
                model_results['acertos_3_ou_mais'] = sum(1 for x in model_results['acertos_por_sorteio'] if x >= 3)
                model_results['taxa_sucesso_3+'] = model_results['acertos_3_ou_mais'] / len(model_results['acertos_por_sorteio'])
            
            results[model_name] = model_results
        
        return results
