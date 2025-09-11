"""
Модуль машинного обучения для прогнозирования отказов и анализа аномалий
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import warnings

from .data_models import PerformanceMetric, FailureEvent, EnvironmentalCondition, EventType, SeverityLevel

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class PredictionResult:
    """Результат прогнозирования"""
    timestamp: datetime
    failure_probability: float
    predicted_event_type: Optional[EventType] = None
    predicted_severity: Optional[SeverityLevel] = None
    confidence: float = 0.0
    features_importance: Optional[Dict[str, float]] = None


@dataclass
class AnomalyResult:
    """Результат обнаружения аномалий"""
    timestamp: datetime
    anomaly_score: float
    is_anomaly: bool
    anomaly_type: str = "unknown"
    confidence: float = 0.0


class MLPredictor:
    """Предиктор на основе машинного обучения"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация предиктора
        
        Args:
            config: Конфигурация ML моделей
        """
        self.config = config or {}
        self.models = {}
        self.feature_columns = []
        self.is_trained = False
        
        # Настройки моделей
        self.failure_model_config = self.config.get('failure_prediction', {})
        self.anomaly_model_config = self.config.get('anomaly_detection', {})
        self.timeseries_model_config = self.config.get('time_series', {})
    
    def prepare_features(self, metrics: List[PerformanceMetric], 
                        conditions: List[EnvironmentalCondition],
                        window_size: int = 24) -> pd.DataFrame:
        """
        Подготовка признаков для обучения ML моделей
        
        Args:
            metrics: Метрики производительности
            conditions: Внешние условия
            window_size: Размер окна для временных признаков
            
        Returns:
            DataFrame с признаками
        """
        try:
            # Создание общего временного индекса
            all_timestamps = set()
            
            for metric in metrics:
                all_timestamps.update(metric.timestamps)
            
            for condition in conditions:
                all_timestamps.update(condition.timestamps)
            
            if not all_timestamps:
                return pd.DataFrame()
            
            # Сортировка временных меток
            sorted_timestamps = sorted(list(all_timestamps))
            
            # Создание DataFrame
            df = pd.DataFrame(index=sorted_timestamps)
            
            # Добавление метрик производительности
            for metric in metrics:
                metric_series = pd.Series(metric.values, index=metric.timestamps)
                metric_series = metric_series.reindex(sorted_timestamps, method='ffill')
                
                # Базовые признаки
                df[f'{metric.name}_value'] = metric_series
                df[f'{metric.name}_mean'] = metric_series.rolling(window=window_size).mean()
                df[f'{metric.name}_std'] = metric_series.rolling(window=window_size).std()
                df[f'{metric.name}_min'] = metric_series.rolling(window=window_size).min()
                df[f'{metric.name}_max'] = metric_series.rolling(window=window_size).max()
                
                # Тренд
                df[f'{metric.name}_trend'] = metric_series.diff()
                df[f'{metric.name}_trend_ma'] = df[f'{metric.name}_trend'].rolling(window=window_size).mean()
            
            # Добавление внешних условий
            for condition in conditions:
                condition_series = pd.Series(condition.values, index=condition.timestamps)
                condition_series = condition_series.reindex(sorted_timestamps, method='ffill')
                
                df[f'{condition.name}_value'] = condition_series
                df[f'{condition.name}_mean'] = condition_series.rolling(window=window_size).mean()
                df[f'{condition.name}_std'] = condition_series.rolling(window=window_size).std()
            
            # Временные признаки
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
            
            # Циклические признаки
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # Заполнение пропущенных значений
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            self.feature_columns = df.columns.tolist()
            return df
            
        except Exception as e:
            logger.error(f"Ошибка подготовки признаков: {e}")
            return pd.DataFrame()
    
    def prepare_targets(self, failure_events: List[FailureEvent], 
                       timestamps: List[datetime]) -> pd.DataFrame:
        """
        Подготовка целевых переменных для обучения
        
        Args:
            failure_events: События отказов
            timestamps: Временные метки
            
        Returns:
            DataFrame с целевыми переменными
        """
        try:
            df = pd.DataFrame(index=timestamps)
            
            # Бинарная переменная отказа
            df['failure'] = 0
            df['failure_type'] = 'None'
            df['failure_severity'] = 'Low'
            
            # Создание временных окон для предсказания
            prediction_windows = [1, 6, 12, 24]  # часы
            
            for window in prediction_windows:
                df[f'failure_in_{window}h'] = 0
            
            # Заполнение целевых переменных
            for event in failure_events:
                event_time = event.timestamp
                
                # Поиск ближайшей временной метки
                time_diffs = [abs((ts - event_time).total_seconds()) for ts in timestamps]
                closest_idx = np.argmin(time_diffs)
                
                if time_diffs[closest_idx] < 3600:  # В пределах часа
                    df.iloc[closest_idx, df.columns.get_loc('failure')] = 1
                    df.iloc[closest_idx, df.columns.get_loc('failure_type')] = event.event_type.value
                    df.iloc[closest_idx, df.columns.get_loc('failure_severity')] = event.severity.value
                    
                    # Заполнение окон предсказания
                    for window in prediction_windows:
                        start_idx = max(0, closest_idx - window)
                        df.iloc[start_idx:closest_idx, df.columns.get_loc(f'failure_in_{window}h')] = 1
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка подготовки целевых переменных: {e}")
            return pd.DataFrame()
    
    def train_failure_prediction_model(self, features_df: pd.DataFrame, 
                                     targets_df: pd.DataFrame) -> bool:
        """
        Обучение модели прогнозирования отказов
        
        Args:
            features_df: DataFrame с признаками
            targets_df: DataFrame с целевыми переменными
            
        Returns:
            True если обучение успешно
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            from sklearn.metrics import classification_report, accuracy_score
            
            # Подготовка данных
            X = features_df.fillna(0)
            y = targets_df['failure_in_24h'].fillna(0)  # Предсказание на 24 часа вперед
            
            # Разделение на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Обучение модели
            model = RandomForestClassifier(
                n_estimators=self.failure_model_config.get('n_estimators', 100),
                max_depth=self.failure_model_config.get('max_depth', 10),
                random_state=42,
                class_weight='balanced'
            )
            
            model.fit(X_train, y_train)
            
            # Оценка качества
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Точность модели прогнозирования отказов: {accuracy:.3f}")
            
            # Сохранение модели
            self.models['failure_prediction'] = model
            
            # Обучение модели классификации типа отказа
            if 'failure_type' in targets_df.columns:
                self._train_failure_type_classifier(X, targets_df)
            
            # Обучение модели классификации критичности
            if 'failure_severity' in targets_df.columns:
                self._train_severity_classifier(X, targets_df)
            
            self.is_trained = True
            return True
            
        except ImportError:
            logger.error("scikit-learn не установлен")
            return False
        except Exception as e:
            logger.error(f"Ошибка обучения модели прогнозирования: {e}")
            return False
    
    def _train_failure_type_classifier(self, X: pd.DataFrame, targets_df: pd.DataFrame) -> None:
        """Обучение классификатора типа отказа"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            
            # Подготовка данных
            failure_mask = targets_df['failure'] == 1
            if failure_mask.sum() < 2:
                return
            
            X_failures = X[failure_mask]
            y_types = targets_df.loc[failure_mask, 'failure_type']
            
            # Кодирование меток
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_types)
            
            # Обучение модели
            type_model = RandomForestClassifier(n_estimators=50, random_state=42)
            type_model.fit(X_failures, y_encoded)
            
            self.models['failure_type'] = (type_model, le)
            
        except Exception as e:
            logger.warning(f"Ошибка обучения классификатора типа отказа: {e}")
    
    def _train_severity_classifier(self, X: pd.DataFrame, targets_df: pd.DataFrame) -> None:
        """Обучение классификатора критичности отказа"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            
            # Подготовка данных
            failure_mask = targets_df['failure'] == 1
            if failure_mask.sum() < 2:
                return
            
            X_failures = X[failure_mask]
            y_severity = targets_df.loc[failure_mask, 'failure_severity']
            
            # Кодирование меток
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_severity)
            
            # Обучение модели
            severity_model = RandomForestClassifier(n_estimators=50, random_state=42)
            severity_model.fit(X_failures, y_encoded)
            
            self.models['failure_severity'] = (severity_model, le)
            
        except Exception as e:
            logger.warning(f"Ошибка обучения классификатора критичности: {e}")
    
    def train_anomaly_detection_model(self, features_df: pd.DataFrame) -> bool:
        """
        Обучение модели обнаружения аномалий
        
        Args:
            features_df: DataFrame с признаками
            
        Returns:
            True если обучение успешно
        """
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            
            # Подготовка данных
            X = features_df.fillna(0)
            
            # Масштабирование
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Обучение модели
            model = IsolationForest(
                contamination=self.anomaly_model_config.get('contamination', 0.1),
                random_state=42
            )
            
            model.fit(X_scaled)
            
            # Сохранение модели и скейлера
            self.models['anomaly_detection'] = (model, scaler)
            
            logger.info("Модель обнаружения аномалий обучена")
            return True
            
        except ImportError:
            logger.error("scikit-learn не установлен")
            return False
        except Exception as e:
            logger.error(f"Ошибка обучения модели обнаружения аномалий: {e}")
            return False
    
    def predict_failures(self, features_df: pd.DataFrame, 
                        prediction_horizon: int = 24) -> List[PredictionResult]:
        """
        Прогнозирование отказов
        
        Args:
            features_df: DataFrame с признаками
            prediction_horizon: Горизонт прогнозирования в часах
            
        Returns:
            Список результатов прогнозирования
        """
        if not self.is_trained or 'failure_prediction' not in self.models:
            logger.warning("Модель прогнозирования не обучена")
            return []
        
        try:
            model = self.models['failure_prediction']
            X = features_df.fillna(0)
            
            # Прогнозирование вероятности отказа
            failure_probs = model.predict_proba(X)[:, 1]
            
            results = []
            for i, (timestamp, prob) in enumerate(zip(features_df.index, failure_probs)):
                if prob > 0.1:  # Порог для вывода предупреждения
                    # Прогнозирование типа отказа
                    predicted_type = None
                    if 'failure_type' in self.models:
                        type_model, le = self.models['failure_type']
                        type_pred = type_model.predict(X.iloc[[i]])
                        predicted_type = EventType(le.inverse_transform(type_pred)[0])
                    
                    # Прогнозирование критичности
                    predicted_severity = None
                    if 'failure_severity' in self.models:
                        severity_model, le = self.models['failure_severity']
                        severity_pred = severity_model.predict(X.iloc[[i]])
                        predicted_severity = SeverityLevel(le.inverse_transform(severity_pred)[0])
                    
                    # Важность признаков
                    feature_importance = dict(zip(
                        self.feature_columns, 
                        model.feature_importances_
                    ))
                    
                    results.append(PredictionResult(
                        timestamp=timestamp,
                        failure_probability=prob,
                        predicted_event_type=predicted_type,
                        predicted_severity=predicted_severity,
                        confidence=prob,
                        features_importance=feature_importance
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка прогнозирования отказов: {e}")
            return []
    
    def detect_anomalies(self, features_df: pd.DataFrame) -> List[AnomalyResult]:
        """
        Обнаружение аномалий
        
        Args:
            features_df: DataFrame с признаками
            
        Returns:
            Список результатов обнаружения аномалий
        """
        if 'anomaly_detection' not in self.models:
            logger.warning("Модель обнаружения аномалий не обучена")
            return []
        
        try:
            model, scaler = self.models['anomaly_detection']
            X = features_df.fillna(0)
            X_scaled = scaler.transform(X)
            
            # Обнаружение аномалий
            anomaly_scores = model.decision_function(X_scaled)
            anomaly_labels = model.predict(X_scaled)
            
            results = []
            for i, (timestamp, score, label) in enumerate(zip(features_df.index, anomaly_scores, anomaly_labels)):
                is_anomaly = label == -1
                confidence = abs(score)
                
                # Определение типа аномалии
                anomaly_type = "unknown"
                if is_anomaly:
                    # Простая эвристика для определения типа
                    if confidence > 0.5:
                        anomaly_type = "severe"
                    else:
                        anomaly_type = "moderate"
                
                results.append(AnomalyResult(
                    timestamp=timestamp,
                    anomaly_score=score,
                    is_anomaly=is_anomaly,
                    anomaly_type=anomaly_type,
                    confidence=confidence
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка обнаружения аномалий: {e}")
            return []
    
    def train_time_series_model(self, metric: PerformanceMetric) -> bool:
        """
        Обучение модели временных рядов для прогнозирования
        
        Args:
            metric: Метрика производительности
            
        Returns:
            True если обучение успешно
        """
        try:
            # Простая ARIMA модель
            from statsmodels.tsa.arima.model import ARIMA
            
            values = np.array(metric.values)
            
            # Автоматический подбор параметров ARIMA
            best_aic = float('inf')
            best_model = None
            
            for p in range(0, 3):
                for d in range(0, 2):
                    for q in range(0, 3):
                        try:
                            model = ARIMA(values, order=(p, d, q))
                            fitted_model = model.fit()
                            
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_model = fitted_model
                        except:
                            continue
            
            if best_model is not None:
                self.models[f'timeseries_{metric.name}'] = best_model
                logger.info(f"Модель временных рядов для {metric.name} обучена (AIC: {best_aic:.2f})")
                return True
            
            return False
            
        except ImportError:
            logger.warning("statsmodels не установлен, временные ряды не поддерживаются")
            return False
        except Exception as e:
            logger.error(f"Ошибка обучения модели временных рядов: {e}")
            return False
    
    def predict_time_series(self, metric_name: str, steps: int = 24) -> List[float]:
        """
        Прогнозирование временного ряда
        
        Args:
            metric_name: Название метрики
            steps: Количество шагов прогноза
            
        Returns:
            Список прогнозируемых значений
        """
        model_key = f'timeseries_{metric_name}'
        
        if model_key not in self.models:
            logger.warning(f"Модель временных рядов для {metric_name} не обучена")
            return []
        
        try:
            model = self.models[model_key]
            forecast = model.forecast(steps=steps)
            return forecast.tolist()
            
        except Exception as e:
            logger.error(f"Ошибка прогнозирования временного ряда: {e}")
            return []
    
    def get_model_performance(self) -> Dict[str, Any]:
        """
        Получение информации о производительности моделей
        
        Returns:
            Словарь с метриками производительности
        """
        performance = {
            'is_trained': self.is_trained,
            'models_count': len(self.models),
            'feature_columns_count': len(self.feature_columns)
        }
        
        for model_name, model in self.models.items():
            if hasattr(model, 'score'):
                performance[f'{model_name}_score'] = model.score
            elif isinstance(model, tuple) and hasattr(model[0], 'score'):
                performance[f'{model_name}_score'] = model[0].score
        
        return performance
