#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тесты для улучшенной системы анализа ИКС
"""

import unittest
import numpy as np
from datetime import datetime, timedelta
from typing import List

from src.analyzer import AdvancedSystemAnalyzer
from src.data_models import EventType, SeverityLevel, PerformanceMetric, FailureEvent
from src.reliability_metrics import ReliabilityCalculator
from src.statistical_analysis import StatisticalAnalyzer
from src.ml_predictions import MLPredictor


class TestAdvancedSystemAnalyzer(unittest.TestCase):
    """Тесты для основного класса анализатора"""
    
    def setUp(self):
        """Настройка тестовых данных"""
        self.analyzer = AdvancedSystemAnalyzer("Test System")
        
        # Создание тестовых данных
        self.timestamps = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
        self.cpu_values = [50 + i + np.random.normal(0, 5) for i in range(24)]
        self.temp_values = [20 + 5 * np.sin(i * np.pi / 12) + np.random.normal(0, 1) for i in range(24)]
        
        # Добавление метрик
        self.analyzer.add_performance_metric(
            name="CPU_Usage",
            values=self.cpu_values,
            timestamps=self.timestamps,
            unit="%",
            threshold_warning=70.0,
            threshold_critical=85.0
        )
        
        # Добавление внешних условий
        self.analyzer.add_environmental_condition(
            name="Temperature",
            values=self.temp_values,
            timestamps=self.timestamps,
            unit="°C",
            normal_range=(15, 30)
        )
        
        # Добавление события отказа
        self.analyzer.add_failure_event(
            timestamp=datetime.now() - timedelta(hours=12),
            event_type=EventType.HARDWARE,
            description="Test failure",
            severity=SeverityLevel.HIGH,
            duration=2.0
        )
    
    def test_add_performance_metric(self):
        """Тест добавления метрики производительности"""
        self.assertEqual(len(self.analyzer.analysis_data.performance_metrics), 1)
        self.assertEqual(self.analyzer.analysis_data.performance_metrics[0].name, "CPU_Usage")
        self.assertEqual(len(self.analyzer.analysis_data.performance_metrics[0].values), 24)
    
    def test_add_failure_event(self):
        """Тест добавления события отказа"""
        self.assertEqual(len(self.analyzer.analysis_data.failure_events), 1)
        self.assertEqual(self.analyzer.analysis_data.failure_events[0].event_type, EventType.HARDWARE)
        self.assertEqual(self.analyzer.analysis_data.failure_events[0].severity, SeverityLevel.HIGH)
    
    def test_calculate_reliability_metrics(self):
        """Тест расчета метрик надежности"""
        metrics = self.analyzer.calculate_reliability_metrics()
        
        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.mtbf, 0)
        self.assertGreater(metrics.mttr, 0)
        self.assertGreaterEqual(metrics.availability, 0)
        self.assertLessEqual(metrics.availability, 1)
        self.assertEqual(metrics.total_failures, 1)
    
    def test_analyze_performance_degradation(self):
        """Тест анализа деградации производительности"""
        degradation = self.analyzer.analyze_performance_degradation()
        
        self.assertEqual(len(degradation), 1)
        self.assertEqual(degradation[0].metric_name, "CPU_Usage")
        self.assertIsInstance(degradation[0].trend_slope, float)
        self.assertIsInstance(degradation[0].is_degrading, bool)
    
    def test_correlate_with_environment(self):
        """Тест корреляционного анализа"""
        correlations = self.analyzer.correlate_with_environment()
        
        self.assertGreater(len(correlations), 0)
        self.assertEqual(correlations[0].metric_name, "CPU_Usage")
        self.assertEqual(correlations[0].condition_name, "Temperature")
        self.assertIsInstance(correlations[0].correlation_coefficient, float)
    
    def test_system_health_score(self):
        """Тест расчета индекса здоровья системы"""
        health_score = self.analyzer.get_system_health_score()
        
        self.assertGreaterEqual(health_score, 0)
        self.assertLessEqual(health_score, 100)
        self.assertIsInstance(health_score, float)


class TestReliabilityCalculator(unittest.TestCase):
    """Тесты для калькулятора метрик надежности"""
    
    def setUp(self):
        """Настройка тестовых данных"""
        self.calculator = ReliabilityCalculator()
        
        # Создание тестовых событий отказов
        base_time = datetime.now()
        self.failure_events = [
            FailureEvent(
                timestamp=base_time - timedelta(hours=48),
                event_type=EventType.HARDWARE,
                description="Test failure 1",
                severity=SeverityLevel.HIGH,
                duration=2.0
            ),
            FailureEvent(
                timestamp=base_time - timedelta(hours=24),
                event_type=EventType.SOFTWARE,
                description="Test failure 2",
                severity=SeverityLevel.MEDIUM,
                duration=1.0
            )
        ]
    
    def test_calculate_metrics(self):
        """Тест расчета метрик надежности"""
        metrics = self.calculator.calculate_metrics(self.failure_events, 72)
        
        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.mtbf, 0)
        self.assertGreater(metrics.mttr, 0)
        self.assertEqual(metrics.total_failures, 2)
        self.assertEqual(metrics.critical_failures, 0)  # Нет критических отказов
    
    def test_perfect_reliability(self):
        """Тест идеальной надежности (без отказов)"""
        metrics = self.calculator.calculate_metrics([], 72)
        
        self.assertEqual(metrics.mtbf, float('inf'))
        self.assertEqual(metrics.mttr, 0.0)
        self.assertEqual(metrics.availability, 1.0)
        self.assertEqual(metrics.total_failures, 0)


class TestStatisticalAnalyzer(unittest.TestCase):
    """Тесты для статистического анализатора"""
    
    def setUp(self):
        """Настройка тестовых данных"""
        self.analyzer = StatisticalAnalyzer()
        
        # Создание тестовой метрики с трендом
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
        values = [50 + i + np.random.normal(0, 2) for i in range(24)]  # Восходящий тренд
        
        self.metric = PerformanceMetric(
            name="Test_Metric",
            values=values,
            timestamps=timestamps,
            unit="%"
        )
    
    def test_analyze_degradation(self):
        """Тест анализа деградации"""
        analysis = self.analyzer.analyze_degradation(self.metric)
        
        self.assertEqual(analysis.metric_name, "Test_Metric")
        self.assertIsInstance(analysis.trend_slope, float)
        self.assertIsInstance(analysis.is_degrading, bool)
        self.assertGreaterEqual(analysis.confidence_level, 0)
        self.assertLessEqual(analysis.confidence_level, 1)
    
    def test_detect_anomalies(self):
        """Тест обнаружения аномалий"""
        # Создание данных с аномалией
        values = [50] * 20 + [100] + [50] * 3  # Аномалия в середине
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
        
        metric = PerformanceMetric(
            name="Test_Metric_Anomaly",
            values=values,
            timestamps=timestamps,
            unit="%"
        )
        
        anomalies = self.analyzer.detect_anomalies(np.array(values))
        
        self.assertIsInstance(anomalies, list)
        # Аномалия должна быть обнаружена
        self.assertGreater(len(anomalies), 0)


class TestMLPredictor(unittest.TestCase):
    """Тесты для ML предиктора"""
    
    def setUp(self):
        """Настройка тестовых данных"""
        self.predictor = MLPredictor()
        
        # Создание тестовых данных
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(48, 0, -1)]
        
        self.metrics = [
            PerformanceMetric(
                name="CPU_Usage",
                values=[50 + i + np.random.normal(0, 5) for i in range(48)],
                timestamps=timestamps,
                unit="%"
            )
        ]
        
        self.conditions = [
            PerformanceMetric(
                name="Temperature",
                values=[20 + 5 * np.sin(i * np.pi / 12) + np.random.normal(0, 1) for i in range(48)],
                timestamps=timestamps,
                unit="°C"
            )
        ]
    
    def test_prepare_features(self):
        """Тест подготовки признаков"""
        features_df = self.predictor.prepare_features(self.metrics, self.conditions)
        
        self.assertFalse(features_df.empty)
        self.assertGreater(len(features_df.columns), 0)
        self.assertEqual(len(features_df), 48)
    
    def test_prepare_targets(self):
        """Тест подготовки целевых переменных"""
        # Создание тестовых событий отказов
        failure_events = [
            FailureEvent(
                timestamp=datetime.now() - timedelta(hours=12),
                event_type=EventType.HARDWARE,
                description="Test failure",
                severity=SeverityLevel.HIGH
            )
        ]
        
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(48, 0, -1)]
        targets_df = self.predictor.prepare_targets(failure_events, timestamps)
        
        self.assertFalse(targets_df.empty)
        self.assertIn('failure', targets_df.columns)
        self.assertEqual(len(targets_df), 48)


def run_performance_test():
    """Тест производительности системы"""
    print("\n" + "="*50)
    print("ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ СИСТЕМЫ")
    print("="*50)
    
    import time
    
    # Создание большого объема данных
    print("Создание тестовых данных...")
    analyzer = AdvancedSystemAnalyzer("Performance Test System")
    
    # 30 дней данных по часам
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(720, 0, -1)]
    
    # Добавление метрик
    start_time = time.time()
    
    for i in range(5):  # 5 метрик
        values = [50 + i * 10 + np.random.normal(0, 5) for _ in range(720)]
        analyzer.add_performance_metric(
            name=f"Metric_{i}",
            values=values,
            timestamps=timestamps,
            unit="%"
        )
    
    # Добавление внешних условий
    for i in range(3):  # 3 внешних условия
        values = [20 + i * 5 + np.random.normal(0, 2) for _ in range(720)]
        analyzer.add_environmental_condition(
            name=f"Condition_{i}",
            values=values,
            timestamps=timestamps,
            unit="°C"
        )
    
    data_creation_time = time.time() - start_time
    print(f"Создание данных: {data_creation_time:.2f} секунд")
    
    # Тест расчета метрик надежности
    start_time = time.time()
    reliability_metrics = analyzer.calculate_reliability_metrics()
    reliability_time = time.time() - start_time
    print(f"Расчет метрик надежности: {reliability_time:.2f} секунд")
    
    # Тест анализа деградации
    start_time = time.time()
    degradation_analysis = analyzer.analyze_performance_degradation()
    degradation_time = time.time() - start_time
    print(f"Анализ деградации: {degradation_time:.2f} секунд")
    
    # Тест корреляционного анализа
    start_time = time.time()
    correlations = analyzer.correlate_with_environment()
    correlation_time = time.time() - start_time
    print(f"Корреляционный анализ: {correlation_time:.2f} секунд")
    
    total_time = data_creation_time + reliability_time + degradation_time + correlation_time
    print(f"Общее время: {total_time:.2f} секунд")
    
    # Проверка производительности
    if total_time < 30:  # Менее 30 секунд для 720 точек данных
        print("✓ Тест производительности ПРОЙДЕН")
    else:
        print("✗ Тест производительности НЕ ПРОЙДЕН")
    
    return total_time < 30


if __name__ == "__main__":
    print("ЗАПУСК ТЕСТОВ УЛУЧШЕННОЙ СИСТЕМЫ АНАЛИЗА ИКС")
    print("="*60)
    
    # Запуск unit тестов
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Запуск теста производительности
    run_performance_test()
    
    print("\n" + "="*60)
    print("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
    print("="*60)
