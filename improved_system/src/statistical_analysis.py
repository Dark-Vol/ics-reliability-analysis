"""
Модуль для статистического анализа деградации производительности
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.signal import find_peaks
import warnings

from .data_models import PerformanceMetric, DegradationAnalysis, EnvironmentalCondition, CorrelationResult

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class StatisticalAnalyzer:
    """Анализатор статистических характеристик системы"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация анализатора
        
        Args:
            config: Конфигурация с параметрами анализа
        """
        self.config = config or {}
        self.trend_window = self.config.get('trend_window', 24)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        self.anomaly_threshold = self.config.get('anomaly_threshold', 3.0)
        self.degradation_threshold = self.config.get('degradation_threshold', -0.1)
    
    def analyze_degradation(self, metric: PerformanceMetric) -> DegradationAnalysis:
        """
        Анализ деградации производительности метрики
        
        Args:
            metric: Метрика производительности
            
        Returns:
            Результат анализа деградации
        """
        values = np.array(metric.values)
        timestamps = metric.timestamps
        
        if len(values) < 3:
            return self._get_empty_degradation_analysis(metric.name)
        
        # Базовые статистики
        mean_value = np.mean(values)
        std_value = np.std(values)
        min_value = np.min(values)
        max_value = np.max(values)
        
        # Коэффициент вариации
        cv = std_value / mean_value if mean_value != 0 else 0
        
        # Анализ тренда
        trend_slope, trend_p_value = self._calculate_trend(values, timestamps)
        
        # Скорость деградации (% в час)
        degradation_rate = self._calculate_degradation_rate(values, timestamps)
        
        # Критические точки (выбросы)
        critical_points = self._count_critical_points(values, mean_value, std_value)
        
        # Определение деградации
        is_degrading = (trend_slope < self.degradation_threshold and 
                       trend_p_value < 0.05)
        
        # Уровень доверия
        confidence_level = 1 - trend_p_value if trend_p_value < 1 else 0
        
        return DegradationAnalysis(
            metric_name=metric.name,
            trend_slope=trend_slope,
            trend_p_value=trend_p_value,
            coefficient_of_variation=cv,
            critical_points=critical_points,
            mean_value=mean_value,
            std_value=std_value,
            min_value=min_value,
            max_value=max_value,
            degradation_rate=degradation_rate,
            is_degrading=is_degrading,
            confidence_level=confidence_level
        )
    
    def _get_empty_degradation_analysis(self, metric_name: str) -> DegradationAnalysis:
        """Возврат пустого анализа для недостаточных данных"""
        return DegradationAnalysis(
            metric_name=metric_name,
            trend_slope=0.0,
            trend_p_value=1.0,
            coefficient_of_variation=0.0,
            critical_points=0,
            mean_value=0.0,
            std_value=0.0,
            min_value=0.0,
            max_value=0.0,
            degradation_rate=0.0,
            is_degrading=False,
            confidence_level=0.0
        )
    
    def _calculate_trend(self, values: np.ndarray, timestamps: List[datetime]) -> Tuple[float, float]:
        """
        Расчет тренда с использованием теста Манна-Кендалла
        
        Args:
            values: Значения метрики
            timestamps: Временные метки
            
        Returns:
            Кортеж (наклон тренда, p-value)
        """
        try:
            # Простой линейный тренд
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            # Тест Манна-Кендалла для проверки значимости тренда
            mk_stat, mk_p_value = self._mann_kendall_test(values)
            
            return slope, mk_p_value
            
        except Exception as e:
            logger.warning(f"Ошибка расчета тренда: {e}")
            return 0.0, 1.0
    
    def _mann_kendall_test(self, values: np.ndarray) -> Tuple[float, float]:
        """
        Тест Манна-Кендалла для выявления тренда
        
        Args:
            values: Временной ряд
            
        Returns:
            Кортеж (статистика, p-value)
        """
        n = len(values)
        if n < 3:
            return 0.0, 1.0
        
        # Подсчет инверсий
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                s += np.sign(values[j] - values[i])
        
        # Расчет статистики
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # p-value (двусторонний тест)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return z, p_value
    
    def _calculate_degradation_rate(self, values: np.ndarray, 
                                   timestamps: List[datetime]) -> float:
        """
        Расчет скорости деградации (% в час)
        
        Args:
            values: Значения метрики
            timestamps: Временные метки
            
        Returns:
            Скорость деградации в %/час
        """
        if len(values) < 2:
            return 0.0
        
        try:
            # Расчет общего времени в часах
            total_hours = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
            
            if total_hours <= 0:
                return 0.0
            
            # Относительное изменение
            relative_change = (values[-1] - values[0]) / abs(values[0]) if values[0] != 0 else 0
            
            # Скорость деградации в %/час
            degradation_rate = (relative_change * 100) / total_hours
            
            return degradation_rate
            
        except Exception as e:
            logger.warning(f"Ошибка расчета скорости деградации: {e}")
            return 0.0
    
    def _count_critical_points(self, values: np.ndarray, mean: float, std: float) -> int:
        """
        Подсчет критических точек (выбросов)
        
        Args:
            values: Значения метрики
            mean: Среднее значение
            std: Стандартное отклонение
            
        Returns:
            Количество критических точек
        """
        if std == 0:
            return 0
        
        # Критерий 3-сигм
        threshold = self.anomaly_threshold * std
        critical_points = np.sum(np.abs(values - mean) > threshold)
        
        return int(critical_points)
    
    def detect_anomalies(self, values: np.ndarray, method: str = 'isolation_forest') -> List[int]:
        """
        Обнаружение аномалий во временном ряду
        
        Args:
            values: Временной ряд
            method: Метод обнаружения ('isolation_forest', 'z_score', 'iqr')
            
        Returns:
            Индексы аномальных точек
        """
        if len(values) < 3:
            return []
        
        try:
            if method == 'z_score':
                return self._z_score_anomalies(values)
            elif method == 'iqr':
                return self._iqr_anomalies(values)
            elif method == 'isolation_forest':
                return self._isolation_forest_anomalies(values)
            else:
                return self._z_score_anomalies(values)
                
        except Exception as e:
            logger.warning(f"Ошибка обнаружения аномалий: {e}")
            return []
    
    def _z_score_anomalies(self, values: np.ndarray) -> List[int]:
        """Обнаружение аномалий по Z-score"""
        z_scores = np.abs(stats.zscore(values))
        anomaly_indices = np.where(z_scores > self.anomaly_threshold)[0]
        return anomaly_indices.tolist()
    
    def _iqr_anomalies(self, values: np.ndarray) -> List[int]:
        """Обнаружение аномалий по межквартильному размаху"""
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        anomaly_indices = np.where((values < lower_bound) | (values > upper_bound))[0]
        return anomaly_indices.tolist()
    
    def _isolation_forest_anomalies(self, values: np.ndarray) -> List[int]:
        """Обнаружение аномалий с помощью Isolation Forest"""
        try:
            from sklearn.ensemble import IsolationForest
            
            # Подготовка данных
            X = values.reshape(-1, 1)
            
            # Обучение модели
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(X)
            
            # Индексы аномалий
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            return anomaly_indices.tolist()
            
        except ImportError:
            logger.warning("scikit-learn не установлен, используется Z-score")
            return self._z_score_anomalies(values)
    
    def calculate_correlation(self, metric: PerformanceMetric, 
                            condition: EnvironmentalCondition) -> CorrelationResult:
        """
        Расчет корреляции между метрикой и внешним условием
        
        Args:
            metric: Метрика производительности
            condition: Внешнее условие
            
        Returns:
            Результат корреляционного анализа
        """
        metric_values = np.array(metric.values)
        condition_values = np.array(condition.values)
        
        # Приведение к одинаковой длине
        min_length = min(len(metric_values), len(condition_values))
        if min_length < 3:
            return self._get_empty_correlation_result(metric.name, condition.name)
        
        metric_values = metric_values[:min_length]
        condition_values = condition_values[:min_length]
        
        try:
            # Расчет корреляции Пирсона
            correlation, p_value = stats.pearsonr(metric_values, condition_values)
            
            # Проверка на значимость
            is_significant = p_value < 0.05 and abs(correlation) > self.correlation_threshold
            
            # Определение силы связи
            abs_corr = abs(correlation)
            if abs_corr < 0.3:
                strength = "weak"
            elif abs_corr < 0.7:
                strength = "moderate"
            else:
                strength = "strong"
            
            return CorrelationResult(
                metric_name=metric.name,
                condition_name=condition.name,
                correlation_coefficient=correlation,
                p_value=p_value,
                significance_level=0.05,
                is_significant=is_significant,
                relationship_strength=strength
            )
            
        except Exception as e:
            logger.warning(f"Ошибка расчета корреляции: {e}")
            return self._get_empty_correlation_result(metric.name, condition.name)
    
    def _get_empty_correlation_result(self, metric_name: str, condition_name: str) -> CorrelationResult:
        """Возврат пустого результата корреляции"""
        return CorrelationResult(
            metric_name=metric_name,
            condition_name=condition_name,
            correlation_coefficient=0.0,
            p_value=1.0,
            significance_level=0.05,
            is_significant=False,
            relationship_strength="weak"
        )
    
    def detect_change_points(self, values: np.ndarray, method: str = 'pettitt') -> List[int]:
        """
        Обнаружение точек изменения тренда
        
        Args:
            values: Временной ряд
            method: Метод обнаружения ('pettitt', 'cpm')
            
        Returns:
            Индексы точек изменения
        """
        if len(values) < 10:
            return []
        
        try:
            if method == 'pettitt':
                return self._pettitt_test(values)
            else:
                return self._pettitt_test(values)
                
        except Exception as e:
            logger.warning(f"Ошибка обнаружения точек изменения: {e}")
            return []
    
    def _pettitt_test(self, values: np.ndarray) -> List[int]:
        """Тест Петтитта для обнаружения точек изменения"""
        n = len(values)
        change_points = []
        
        for t in range(1, n - 1):
            # Статистика для точки t
            u_t = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if j <= t:
                        u_t += np.sign(values[i] - values[j])
                    else:
                        u_t -= np.sign(values[i] - values[j])
            
            # Критическое значение (упрощенное)
            if abs(u_t) > 1.96 * np.sqrt(n * (n - 1) * (2 * n + 5) / 18):
                change_points.append(t)
        
        return change_points
    
    def calculate_seasonality(self, values: np.ndarray, period: int = 24) -> Dict[str, float]:
        """
        Анализ сезонности временного ряда
        
        Args:
            values: Временной ряд
            period: Период сезонности (в часах)
            
        Returns:
            Словарь с характеристиками сезонности
        """
        if len(values) < 2 * period:
            return {'seasonal_strength': 0.0, 'trend_strength': 0.0, 'noise_strength': 0.0}
        
        try:
            # Декомпозиция временного ряда
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Создание временного ряда
            ts = pd.Series(values, index=pd.date_range('2023-01-01', periods=len(values), freq='H'))
            
            # Декомпозиция
            decomposition = seasonal_decompose(ts, model='additive', period=period)
            
            # Расчет силы компонентов
            seasonal_strength = np.var(decomposition.seasonal) / np.var(values)
            trend_strength = np.var(decomposition.trend) / np.var(values)
            noise_strength = np.var(decomposition.resid) / np.var(values)
            
            return {
                'seasonal_strength': seasonal_strength,
                'trend_strength': trend_strength,
                'noise_strength': noise_strength
            }
            
        except ImportError:
            logger.warning("statsmodels не установлен, сезонность не анализируется")
            return {'seasonal_strength': 0.0, 'trend_strength': 0.0, 'noise_strength': 0.0}
        except Exception as e:
            logger.warning(f"Ошибка анализа сезонности: {e}")
            return {'seasonal_strength': 0.0, 'trend_strength': 0.0, 'noise_strength': 0.0}
