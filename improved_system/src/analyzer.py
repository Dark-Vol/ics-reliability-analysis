"""
Улучшенный анализатор функционирования ИКС в неблагоприятных условиях
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

from .config import Config
from .data_models import (
    SystemAnalysisData, PerformanceMetric, FailureEvent, EnvironmentalCondition,
    ReliabilityMetrics, DegradationAnalysis, CorrelationResult, EventType, SeverityLevel
)
from .reliability_metrics import ReliabilityCalculator
from .statistical_analysis import StatisticalAnalyzer
from .ml_predictions import MLPredictor, PredictionResult, AnomalyResult

logger = logging.getLogger(__name__)


class AdvancedSystemAnalyzer:
    """
    Улучшенный анализатор функционирования информационно-коммуникационных систем
    в неблагоприятных условиях с поддержкой машинного обучения и расширенной аналитики
    """
    
    def __init__(self, system_name: str, config_path: Optional[str] = None):
        """
        Инициализация анализатора
        
        Args:
            system_name: Название анализируемой системы
            config_path: Путь к файлу конфигурации
        """
        self.system_name = system_name
        self.config = Config(config_path)
        
        # Инициализация компонентов
        self.reliability_calculator = ReliabilityCalculator(self.config.get('reliability', {}))
        self.statistical_analyzer = StatisticalAnalyzer(self.config.get('analysis', {}))
        self.ml_predictor = MLPredictor(self.config.get('ml', {}))
        
        # Данные системы
        self.analysis_data = SystemAnalysisData(
            system_name=system_name,
            analysis_timestamp=datetime.now()
        )
        
        # Настройка логирования
        self._setup_logging()
        
        logger.info(f"Инициализирован анализатор для системы: {system_name}")
    
    def _setup_logging(self) -> None:
        """Настройка системы логирования"""
        log_level = getattr(logging, self.config.get('system.log_level', 'INFO'))
        log_file = self.config.get('system.log_file', 'system_analysis.log')
        
        # Настройка логгера
        logger.setLevel(log_level)
        
        # Обработчик файла
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        
        # Обработчик консоли
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Форматтер
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Добавление обработчиков
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
    
    def add_performance_metric(self, name: str, values: List[float], 
                             timestamps: List[datetime], unit: str = "",
                             description: str = "", threshold_warning: Optional[float] = None,
                             threshold_critical: Optional[float] = None) -> None:
        """
        Добавление метрики производительности системы
        
        Args:
            name: Название метрики
            values: Значения метрики
            timestamps: Временные метки
            unit: Единица измерения
            description: Описание метрики
            threshold_warning: Порог предупреждения
            threshold_critical: Критический порог
        """
        try:
            metric = PerformanceMetric(
                name=name,
                values=values,
                timestamps=timestamps,
                unit=unit,
                description=description,
                threshold_warning=threshold_warning,
                threshold_critical=threshold_critical
            )
            
            self.analysis_data.performance_metrics.append(metric)
            logger.info(f"Добавлена метрика '{name}' с {len(values)} значениями")
            
        except Exception as e:
            logger.error(f"Ошибка добавления метрики '{name}': {e}")
    
    def add_failure_event(self, timestamp: datetime, event_type: EventType,
                         description: str, severity: SeverityLevel,
                         duration: Optional[float] = None,
                         recovery_time: Optional[datetime] = None,
                         affected_components: Optional[List[str]] = None,
                         root_cause: Optional[str] = None,
                         resolution: Optional[str] = None) -> None:
        """
        Добавление события отказа системы
        
        Args:
            timestamp: Время события
            event_type: Тип события
            description: Описание события
            severity: Критичность события
            duration: Длительность отказа в часах
            recovery_time: Время восстановления
            affected_components: Затронутые компоненты
            root_cause: Причина отказа
            resolution: Способ устранения
        """
        try:
            event = FailureEvent(
                timestamp=timestamp,
                event_type=event_type,
                description=description,
                severity=severity,
                duration=duration,
                recovery_time=recovery_time,
                affected_components=affected_components or [],
                root_cause=root_cause,
                resolution=resolution
            )
            
            self.analysis_data.failure_events.append(event)
            logger.warning(f"Зарегистрирован отказ: {event_type.value} - {description}")
            
        except Exception as e:
            logger.error(f"Ошибка добавления события отказа: {e}")
    
    def add_environmental_condition(self, name: str, values: List[float],
                                   timestamps: List[datetime], unit: str = "",
                                   description: str = "", normal_range: Optional[tuple] = None,
                                   critical_range: Optional[tuple] = None) -> None:
        """
        Добавление данных о внешних условиях
        
        Args:
            name: Название условия
            values: Значения условия
            timestamps: Временные метки
            unit: Единица измерения
            description: Описание условия
            normal_range: Нормальный диапазон (min, max)
            critical_range: Критический диапазон (min, max)
        """
        try:
            condition = EnvironmentalCondition(
                name=name,
                values=values,
                timestamps=timestamps,
                unit=unit,
                description=description,
                normal_range=normal_range,
                critical_range=critical_range
            )
            
            self.analysis_data.environmental_conditions.append(condition)
            logger.info(f"Добавлено условие '{name}' с {len(values)} значениями")
            
        except Exception as e:
            logger.error(f"Ошибка добавления внешнего условия '{name}': {e}")
    
    def calculate_reliability_metrics(self, analysis_period_hours: float = 720) -> ReliabilityMetrics:
        """
        Расчет расширенных метрик надежности системы
        
        Args:
            analysis_period_hours: Период анализа в часах
            
        Returns:
            Объект с метриками надежности
        """
        try:
            metrics = self.reliability_calculator.calculate_metrics(
                self.analysis_data.failure_events,
                analysis_period_hours
            )
            
            self.analysis_data.reliability_metrics = metrics
            logger.info("Рассчитаны метрики надежности")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Ошибка расчета метрик надежности: {e}")
            return ReliabilityMetrics(
                mtbf=float('inf'), mttr=0.0, mttf=analysis_period_hours,
                availability=1.0, rto=0.0, rpo=0.0, total_failures=0,
                critical_failures=0, uptime=analysis_period_hours, downtime=0.0
            )
    
    def analyze_performance_degradation(self) -> List[DegradationAnalysis]:
        """
        Расширенный анализ деградации производительности системы
        
        Returns:
            Список результатов анализа деградации
        """
        try:
            degradation_results = []
            
            for metric in self.analysis_data.performance_metrics:
                analysis = self.statistical_analyzer.analyze_degradation(metric)
                degradation_results.append(analysis)
            
            self.analysis_data.degradation_analysis = degradation_results
            logger.info("Проведен анализ деградации производительности")
            
            return degradation_results
            
        except Exception as e:
            logger.error(f"Ошибка анализа деградации: {e}")
            return []
    
    def correlate_with_environment(self) -> List[CorrelationResult]:
        """
        Расширенный корреляционный анализ между производительностью и внешними условиями
        
        Returns:
            Список результатов корреляционного анализа
        """
        try:
            correlation_results = []
            
            for metric in self.analysis_data.performance_metrics:
                for condition in self.analysis_data.environmental_conditions:
                    correlation = self.statistical_analyzer.calculate_correlation(metric, condition)
                    correlation_results.append(correlation)
            
            self.analysis_data.correlation_results = correlation_results
            logger.info(f"Рассчитаны корреляции: {len(correlation_results)} пар")
            
            return correlation_results
            
        except Exception as e:
            logger.error(f"Ошибка корреляционного анализа: {e}")
            return []
    
    def train_ml_models(self) -> bool:
        """
        Обучение моделей машинного обучения
        
        Returns:
            True если обучение успешно
        """
        try:
            if not self.analysis_data.performance_metrics:
                logger.warning("Нет данных для обучения ML моделей")
                return False
            
            # Подготовка признаков
            features_df = self.ml_predictor.prepare_features(
                self.analysis_data.performance_metrics,
                self.analysis_data.environmental_conditions
            )
            
            if features_df.empty:
                logger.warning("Не удалось подготовить признаки для ML")
                return False
            
            # Подготовка целевых переменных
            all_timestamps = features_df.index.tolist()
            targets_df = self.ml_predictor.prepare_targets(
                self.analysis_data.failure_events,
                all_timestamps
            )
            
            # Обучение моделей
            success = True
            
            # Модель прогнозирования отказов
            if not self.ml_predictor.train_failure_prediction_model(features_df, targets_df):
                success = False
            
            # Модель обнаружения аномалий
            if not self.ml_predictor.train_anomaly_detection_model(features_df):
                success = False
            
            # Модели временных рядов для каждой метрики
            for metric in self.analysis_data.performance_metrics:
                if not self.ml_predictor.train_time_series_model(metric):
                    logger.warning(f"Не удалось обучить модель временных рядов для {metric.name}")
            
            if success:
                logger.info("ML модели успешно обучены")
            else:
                logger.warning("Некоторые ML модели не удалось обучить")
            
            return success
            
        except Exception as e:
            logger.error(f"Ошибка обучения ML моделей: {e}")
            return False
    
    def predict_failures(self, prediction_horizon: int = 24) -> List[PredictionResult]:
        """
        Прогнозирование отказов системы
        
        Args:
            prediction_horizon: Горизонт прогнозирования в часах
            
        Returns:
            Список результатов прогнозирования
        """
        try:
            if not self.ml_predictor.is_trained:
                logger.warning("ML модели не обучены")
                return []
            
            # Подготовка признаков
            features_df = self.ml_predictor.prepare_features(
                self.analysis_data.performance_metrics,
                self.analysis_data.environmental_conditions
            )
            
            if features_df.empty:
                logger.warning("Не удалось подготовить признаки для прогнозирования")
                return []
            
            # Прогнозирование
            predictions = self.ml_predictor.predict_failures(features_df, prediction_horizon)
            
            logger.info(f"Получено {len(predictions)} прогнозов отказов")
            return predictions
            
        except Exception as e:
            logger.error(f"Ошибка прогнозирования отказов: {e}")
            return []
    
    def detect_anomalies(self) -> List[AnomalyResult]:
        """
        Обнаружение аномалий в работе системы
        
        Returns:
            Список результатов обнаружения аномалий
        """
        try:
            if not self.ml_predictor.is_trained:
                logger.warning("ML модели не обучены")
                return []
            
            # Подготовка признаков
            features_df = self.ml_predictor.prepare_features(
                self.analysis_data.performance_metrics,
                self.analysis_data.environmental_conditions
            )
            
            if features_df.empty:
                logger.warning("Не удалось подготовить признаки для обнаружения аномалий")
                return []
            
            # Обнаружение аномалий
            anomalies = self.ml_predictor.detect_anomalies(features_df)
            
            logger.info(f"Обнаружено {len(anomalies)} аномалий")
            return anomalies
            
        except Exception as e:
            logger.error(f"Ошибка обнаружения аномалий: {e}")
            return []
    
    def generate_comprehensive_report(self) -> str:
        """
        Генерация расширенного отчета о функционировании системы
        
        Returns:
            Текстовый отчет
        """
        try:
            report = f"""
=== РАСШИРЕННЫЙ ОТЧЕТ О ФУНКЦИОНИРОВАНИИ СИСТЕМЫ ===
Система: {self.system_name}
Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Версия анализатора: {self.config.get('system.version', '2.0.0')}

1. МЕТРИКИ НАДЕЖНОСТИ:
"""
            
            # Метрики надежности
            if self.analysis_data.reliability_metrics:
                metrics = self.analysis_data.reliability_metrics
                report += f"   MTBF (среднее время между отказами): {metrics.mtbf:.2f} часов\n"
                report += f"   MTTR (среднее время восстановления): {metrics.mttr:.2f} часов\n"
                report += f"   MTTF (среднее время до отказа): {metrics.mttf:.2f} часов\n"
                report += f"   Доступность: {metrics.availability:.4f} ({metrics.availability*100:.2f}%)\n"
                report += f"   RTO (целевое время восстановления): {metrics.rto:.2f} часов\n"
                report += f"   RPO (целевая точка восстановления): {metrics.rpo:.2f} часов\n"
                report += f"   Общее количество отказов: {metrics.total_failures}\n"
                report += f"   Критических отказов: {metrics.critical_failures}\n"
                report += f"   Время работы: {metrics.uptime:.2f} часов\n"
                report += f"   Время простоя: {metrics.downtime:.2f} часов\n"
            
            # Анализ деградации
            report += "\n2. АНАЛИЗ ДЕГРАДАЦИИ ПРОИЗВОДИТЕЛЬНОСТИ:\n"
            for analysis in self.analysis_data.degradation_analysis:
                report += f"   {analysis.metric_name}:\n"
                report += f"     Тренд: {analysis.trend_slope:.6f} (p-value: {analysis.trend_p_value:.4f})\n"
                report += f"     Коэффициент вариации: {analysis.coefficient_of_variation:.4f}\n"
                report += f"     Критические точки: {analysis.critical_points}\n"
                report += f"     Скорость деградации: {analysis.degradation_rate:.4f} %/час\n"
                report += f"     Деградация: {'ДА' if analysis.is_degrading else 'НЕТ'}\n"
                report += f"     Уровень доверия: {analysis.confidence_level:.4f}\n"
            
            # Корреляции
            report += "\n3. КОРРЕЛЯЦИИ С ВНЕШНИМИ УСЛОВИЯМИ:\n"
            for corr in self.analysis_data.correlation_results:
                if corr.is_significant:
                    report += f"   {corr.metric_name} vs {corr.condition_name}:\n"
                    report += f"     Корреляция: {corr.correlation_coefficient:.4f}\n"
                    report += f"     p-value: {corr.p_value:.4f}\n"
                    report += f"     Сила связи: {corr.relationship_strength}\n"
            
            # События отказов
            report += f"\n4. СОБЫТИЯ ОТКАЗОВ ({len(self.analysis_data.failure_events)}):\n"
            for i, event in enumerate(self.analysis_data.failure_events[-10:], 1):  # Последние 10 событий
                report += f"   {i}. {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {event.event_type.value}\n"
                report += f"      Критичность: {event.severity.value}\n"
                report += f"      Описание: {event.description}\n"
                if event.duration:
                    report += f"      Длительность: {event.duration:.2f} часов\n"
                if event.root_cause:
                    report += f"      Причина: {event.root_cause}\n"
                report += "\n"
            
            # ML модели
            if self.ml_predictor.is_trained:
                report += "\n5. МОДЕЛИ МАШИННОГО ОБУЧЕНИЯ:\n"
                performance = self.ml_predictor.get_model_performance()
                report += f"   Статус: Обучены\n"
                report += f"   Количество моделей: {performance['models_count']}\n"
                report += f"   Количество признаков: {performance['feature_columns_count']}\n"
            
            return report
            
        except Exception as e:
            logger.error(f"Ошибка генерации отчета: {e}")
            return f"Ошибка генерации отчета: {e}"
    
    def save_analysis_data(self, filename: str) -> None:
        """
        Сохранение данных анализа в JSON файл
        
        Args:
            filename: Имя файла для сохранения
        """
        try:
            self.analysis_data.save_to_json(filename)
            logger.info(f"Данные анализа сохранены в файл: {filename}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения данных: {e}")
    
    def load_analysis_data(self, filename: str) -> bool:
        """
        Загрузка данных анализа из JSON файла
        
        Args:
            filename: Имя файла для загрузки
            
        Returns:
            True если загрузка успешна
        """
        try:
            self.analysis_data = SystemAnalysisData.load_from_json(filename)
            logger.info(f"Данные анализа загружены из файла: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            return False
    
    def export_to_excel(self, filename: str) -> None:
        """
        Экспорт данных анализа в Excel файл
        
        Args:
            filename: Имя файла для экспорта
        """
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Метрики производительности
                if self.analysis_data.performance_metrics:
                    metrics_data = []
                    for metric in self.analysis_data.performance_metrics:
                        for i, (value, timestamp) in enumerate(zip(metric.values, metric.timestamps)):
                            metrics_data.append({
                                'Metric': metric.name,
                                'Timestamp': timestamp,
                                'Value': value,
                                'Unit': metric.unit
                            })
                    
                    df_metrics = pd.DataFrame(metrics_data)
                    df_metrics.to_excel(writer, sheet_name='Performance_Metrics', index=False)
                
                # События отказов
                if self.analysis_data.failure_events:
                    events_data = [event.to_dict() for event in self.analysis_data.failure_events]
                    df_events = pd.DataFrame(events_data)
                    df_events.to_excel(writer, sheet_name='Failure_Events', index=False)
                
                # Внешние условия
                if self.analysis_data.environmental_conditions:
                    conditions_data = []
                    for condition in self.analysis_data.environmental_conditions:
                        for i, (value, timestamp) in enumerate(zip(condition.values, condition.timestamps)):
                            conditions_data.append({
                                'Condition': condition.name,
                                'Timestamp': timestamp,
                                'Value': value,
                                'Unit': condition.unit
                            })
                    
                    df_conditions = pd.DataFrame(conditions_data)
                    df_conditions.to_excel(writer, sheet_name='Environmental_Conditions', index=False)
                
                # Метрики надежности
                if self.analysis_data.reliability_metrics:
                    df_reliability = pd.DataFrame([self.analysis_data.reliability_metrics.to_dict()])
                    df_reliability.to_excel(writer, sheet_name='Reliability_Metrics', index=False)
            
            logger.info(f"Данные экспортированы в Excel файл: {filename}")
            
        except ImportError:
            logger.error("openpyxl не установлен, экспорт в Excel недоступен")
        except Exception as e:
            logger.error(f"Ошибка экспорта в Excel: {e}")
    
    def get_system_health_score(self) -> float:
        """
        Расчет общего индекса здоровья системы (0-100)
        
        Returns:
            Индекс здоровья системы
        """
        try:
            score = 100.0
            
            # Штраф за отказы
            if self.analysis_data.reliability_metrics:
                availability = self.analysis_data.reliability_metrics.availability
                score *= availability
                
                # Дополнительный штраф за критические отказы
                critical_penalty = self.analysis_data.reliability_metrics.critical_failures * 5
                score -= critical_penalty
            
            # Штраф за деградацию
            for analysis in self.analysis_data.degradation_analysis:
                if analysis.is_degrading:
                    degradation_penalty = abs(analysis.degradation_rate) * 10
                    score -= degradation_penalty
            
            # Штраф за аномалии
            anomalies = self.detect_anomalies()
            anomaly_penalty = len(anomalies) * 2
            score -= anomaly_penalty
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error(f"Ошибка расчета индекса здоровья: {e}")
            return 0.0
