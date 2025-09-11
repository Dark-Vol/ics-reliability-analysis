#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Улучшенная версия анализа функционирования информационно-коммуникационных систем 
в неблагоприятных условиях с отображением графиков

Версия 2.0 - с поддержкой машинного обучения, расширенной аналитики и модульной архитектуры
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging
import warnings
from pathlib import Path

# Импорт модулей улучшенной системы
import sys
sys.path.append('improved_system')
from src.config import Config
from src.analyzer import AdvancedSystemAnalyzer
from src.data_models import EventType, SeverityLevel
from src.visualization import AdvancedVisualizer

# Подавление предупреждений
warnings.filterwarnings('ignore')

# Настройка pandas для корректного отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Настройка matplotlib для отображения графиков
import matplotlib
matplotlib.use('TkAgg')  # Используем TkAgg backend для отображения
import matplotlib.pyplot as plt
plt.ion()  # Включаем интерактивный режим

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('improved_system/improved_system_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_realistic_sample_data() -> AdvancedSystemAnalyzer:
    """
    Создание реалистичных тестовых данных для демонстрации улучшенной системы
    
    Returns:
        Настроенный анализатор с тестовыми данными
    """
    print("Создание реалистичных тестовых данных...")
    
    # Инициализация анализатора
    analyzer = AdvancedSystemAnalyzer("Промышленная ИКС v2.0", "improved_system/config.yaml")
    
    # Генерация тестовых данных
    np.random.seed(42)
    start_time = datetime.now() - timedelta(days=30)
    
    # Создание временных меток (30 дней по часам)
    timestamps = [start_time + timedelta(hours=i) for i in range(720)]
    
    # 1. Метрики производительности с реалистичными паттернами
    
    # CPU загрузка с трендом деградации и сезонностью
    cpu_base = 40
    cpu_trend = np.linspace(0, 15, 720)  # Деградация
    cpu_seasonal = 10 * np.sin(np.linspace(0, 4*np.pi, 720))  # Сезонность
    cpu_noise = np.random.normal(0, 3, 720)
    cpu_values = cpu_base + cpu_trend + cpu_seasonal + cpu_noise
    cpu_values = np.clip(cpu_values, 0, 100)
    
    analyzer.add_performance_metric(
        name="CPU_Usage",
        values=cpu_values.tolist(),
        timestamps=timestamps,
        unit="%",
        description="Загрузка центрального процессора",
        threshold_warning=70.0,
        threshold_critical=85.0
    )
    
    # Пропускная способность сети с деградацией
    throughput_base = 1000
    throughput_trend = -np.linspace(0, 200, 720)  # Деградация
    throughput_seasonal = 100 * np.sin(np.linspace(0, 2*np.pi, 720))
    throughput_noise = np.random.normal(0, 20, 720)
    throughput_values = throughput_base + throughput_trend + throughput_seasonal + throughput_noise
    throughput_values = np.clip(throughput_values, 0, 1500)
    
    analyzer.add_performance_metric(
        name="Network_Throughput",
        values=throughput_values.tolist(),
        timestamps=timestamps,
        unit="Mbps",
        description="Пропускная способность сети",
        threshold_warning=800.0,
        threshold_critical=600.0
    )
    
    # Использование памяти
    memory_base = 60
    memory_trend = np.linspace(0, 20, 720)
    memory_seasonal = 5 * np.sin(np.linspace(0, 6*np.pi, 720))
    memory_noise = np.random.normal(0, 2, 720)
    memory_values = memory_base + memory_trend + memory_seasonal + memory_noise
    memory_values = np.clip(memory_values, 0, 100)
    
    analyzer.add_performance_metric(
        name="Memory_Usage",
        values=memory_values.tolist(),
        timestamps=timestamps,
        unit="%",
        description="Использование оперативной памяти",
        threshold_warning=80.0,
        threshold_critical=90.0
    )
    
    # 2. Внешние условия с реалистичными паттернами
    
    # Температура с суточными и сезонными циклами
    temp_base = 22
    temp_daily = 8 * np.sin(np.linspace(0, 30*np.pi, 720))  # Суточный цикл
    temp_seasonal = 3 * np.sin(np.linspace(0, np.pi, 720))  # Сезонный тренд
    temp_noise = np.random.normal(0, 1.5, 720)
    temp_values = temp_base + temp_daily + temp_seasonal + temp_noise
    
    analyzer.add_environmental_condition(
        name="Temperature",
        values=temp_values.tolist(),
        timestamps=timestamps,
        unit="°C",
        description="Температура окружающей среды",
        normal_range=(15, 30),
        critical_range=(5, 40)
    )
    
    # Влажность
    humidity_base = 50
    humidity_daily = 20 * np.sin(np.linspace(0, 30*np.pi, 720) + np.pi/2)
    humidity_noise = np.random.normal(0, 3, 720)
    humidity_values = humidity_base + humidity_daily + humidity_noise
    humidity_values = np.clip(humidity_values, 0, 100)
    
    analyzer.add_environmental_condition(
        name="Humidity",
        values=humidity_values.tolist(),
        timestamps=timestamps,
        unit="%",
        description="Относительная влажность воздуха",
        normal_range=(30, 70),
        critical_range=(10, 90)
    )
    
    # Вибрация (влияет на надежность)
    vibration_base = 0.5
    vibration_trend = np.linspace(0, 0.3, 720)  # Увеличение вибрации
    vibration_noise = np.random.exponential(0.1, 720)
    vibration_values = vibration_base + vibration_trend + vibration_noise
    
    analyzer.add_environmental_condition(
        name="Vibration",
        values=vibration_values.tolist(),
        timestamps=timestamps,
        unit="g",
        description="Уровень вибрации",
        normal_range=(0, 1.0),
        critical_range=(0, 2.0)
    )
    
    # 3. События отказов с реалистичными паттернами
    
    # Отказы оборудования (связаны с температурой и вибрацией)
    failure_times = []
    failure_types = []
    failure_descriptions = []
    failure_severities = []
    
    # Критический отказ из-за перегрева
    failure_times.append(start_time + timedelta(days=7, hours=14))
    failure_types.append(EventType.HARDWARE)
    failure_descriptions.append("Отказ процессора из-за перегрева - температура превысила 45°C")
    failure_severities.append(SeverityLevel.CRITICAL)
    
    # Отказ сети
    failure_times.append(start_time + timedelta(days=14, hours=9))
    failure_types.append(EventType.NETWORK)
    failure_descriptions.append("Потеря связи с удаленным узлом - обрыв кабеля")
    failure_severities.append(SeverityLevel.HIGH)
    
    # Отказ ПО
    failure_times.append(start_time + timedelta(days=21, hours=16))
    failure_types.append(EventType.SOFTWARE)
    failure_descriptions.append("Ошибка в модуле обработки данных - переполнение буфера")
    failure_severities.append(SeverityLevel.MEDIUM)
    
    # Отказ питания
    failure_times.append(start_time + timedelta(days=25, hours=11))
    failure_types.append(EventType.POWER)
    failure_descriptions.append("Сбой питания в критическом узле - отключение ИБП")
    failure_severities.append(SeverityLevel.CRITICAL)
    
    # Отказ из-за вибрации
    failure_times.append(start_time + timedelta(days=28, hours=13))
    failure_types.append(EventType.ENVIRONMENTAL)
    failure_descriptions.append("Отказ жесткого диска из-за повышенной вибрации")
    failure_severities.append(SeverityLevel.HIGH)
    
    # Добавление событий отказов с реалистичными временами восстановления
    recovery_times = [4.5, 2.0, 1.5, 6.0, 3.0]  # часы
    
    for i, (time, ftype, desc, severity) in enumerate(zip(failure_times, failure_types, 
                                                          failure_descriptions, failure_severities)):
        recovery_time = time + timedelta(hours=recovery_times[i])
        
        analyzer.add_failure_event(
            timestamp=time,
            event_type=ftype,
            description=desc,
            severity=severity,
            duration=recovery_times[i],
            recovery_time=recovery_time,
            affected_components=["CPU", "Network", "Storage"][i % 3],
            root_cause=["Перегрев", "Физическое повреждение", "Ошибка ПО", 
                       "Отказ ИБП", "Механическое воздействие"][i],
            resolution=["Замена процессора", "Восстановление кабеля", "Обновление ПО",
                       "Замена ИБП", "Замена жесткого диска"][i]
        )
    
    print(f"Создано {len(analyzer.analysis_data.performance_metrics)} метрик производительности")
    print(f"Создано {len(analyzer.analysis_data.environmental_conditions)} внешних условий")
    print(f"Создано {len(analyzer.analysis_data.failure_events)} событий отказов")
    
    return analyzer


def demonstrate_advanced_analysis(analyzer: AdvancedSystemAnalyzer) -> None:
    """
    Демонстрация расширенных возможностей анализа
    
    Args:
        analyzer: Настроенный анализатор
    """
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ РАСШИРЕННЫХ ВОЗМОЖНОСТЕЙ АНАЛИЗА")
    print("="*60)
    
    # 1. Расчет метрик надежности
    print("\n1. Расчет расширенных метрик надежности...")
    reliability_metrics = analyzer.calculate_reliability_metrics()
    
    print(f"   MTBF: {reliability_metrics.mtbf:.2f} часов")
    print(f"   MTTR: {reliability_metrics.mttr:.2f} часов")
    print(f"   MTTF: {reliability_metrics.mttf:.2f} часов")
    print(f"   Доступность: {reliability_metrics.availability:.4f} ({reliability_metrics.availability*100:.2f}%)")
    print(f"   RTO: {reliability_metrics.rto:.2f} часов")
    print(f"   RPO: {reliability_metrics.rpo:.2f} часов")
    print(f"   Общее количество отказов: {reliability_metrics.total_failures}")
    print(f"   Критических отказов: {reliability_metrics.critical_failures}")
    
    # 2. Анализ деградации производительности
    print("\n2. Анализ деградации производительности...")
    degradation_analysis = analyzer.analyze_performance_degradation()
    
    for analysis in degradation_analysis:
        print(f"   {analysis.metric_name}:")
        print(f"     Тренд: {analysis.trend_slope:.6f} (p-value: {analysis.trend_p_value:.4f})")
        print(f"     Скорость деградации: {analysis.degradation_rate:.4f} %/час")
        print(f"     Деградация: {'ДА' if analysis.is_degrading else 'НЕТ'}")
        print(f"     Критические точки: {analysis.critical_points}")
    
    # 3. Корреляционный анализ
    print("\n3. Корреляционный анализ...")
    correlations = analyzer.correlate_with_environment()
    
    significant_correlations = [c for c in correlations if c.is_significant]
    print(f"   Найдено {len(significant_correlations)} значимых корреляций:")
    
    for corr in significant_correlations:
        print(f"     {corr.metric_name} vs {corr.condition_name}: {corr.correlation_coefficient:.4f} ({corr.relationship_strength})")
    
    # 4. Обучение ML моделей
    print("\n4. Обучение моделей машинного обучения...")
    ml_success = analyzer.train_ml_models()
    
    if ml_success:
        print("   ✓ ML модели успешно обучены")
        
        # 5. Прогнозирование отказов
        print("\n5. Прогнозирование отказов...")
        predictions = analyzer.predict_failures(prediction_horizon=24)
        
        if predictions:
            print(f"   Найдено {len(predictions)} потенциальных отказов:")
            for pred in predictions[:3]:  # Показываем первые 3
                print(f"     {pred.timestamp}: вероятность {pred.failure_probability:.3f}")
                if pred.predicted_event_type:
                    print(f"       Тип: {pred.predicted_event_type.value}")
                if pred.predicted_severity:
                    print(f"       Критичность: {pred.predicted_severity.value}")
        else:
            print("   Потенциальных отказов не обнаружено")
        
        # 6. Обнаружение аномалий
        print("\n6. Обнаружение аномалий...")
        anomalies = analyzer.detect_anomalies()
        
        anomaly_count = len([a for a in anomalies if a.is_anomaly])
        print(f"   Обнаружено {anomaly_count} аномалий")
        
        if anomaly_count > 0:
            for anomaly in anomalies[:3]:  # Показываем первые 3
                if anomaly.is_anomaly:
                    print(f"     {anomaly.timestamp}: {anomaly.anomaly_type} (уверенность: {anomaly.confidence:.3f})")
    else:
        print("   ✗ Ошибка обучения ML моделей")
    
    # 7. Расчет индекса здоровья системы
    print("\n7. Индекс здоровья системы...")
    health_score = analyzer.get_system_health_score()
    print(f"   Индекс здоровья: {health_score:.1f}/100")
    
    if health_score >= 90:
        status = "Отличное"
    elif health_score >= 70:
        status = "Хорошее"
    elif health_score >= 50:
        status = "Удовлетворительное"
    else:
        status = "Критическое"
    
    print(f"   Статус системы: {status}")


def create_advanced_visualizations(analyzer: AdvancedSystemAnalyzer) -> None:
    """
    Создание расширенных визуализаций с отображением графиков
    
    Args:
        analyzer: Настроенный анализатор
    """
    print("\n8. Создание расширенных визуализаций...")
    
    try:
        # Инициализация визуализатора
        visualizer = AdvancedVisualizer(analyzer.config.get('visualization', {}))
        
        # Комплексный дашборд
        print("   Создание комплексного дашборда...")
        dashboard_fig = visualizer.create_comprehensive_dashboard(
            metrics=analyzer.analysis_data.performance_metrics,
            events=analyzer.analysis_data.failure_events,
            conditions=analyzer.analysis_data.environmental_conditions,
            reliability=analyzer.analysis_data.reliability_metrics,
            degradation=analyzer.analysis_data.degradation_analysis,
            correlations=analyzer.analysis_data.correlation_results
        )
        
        visualizer.save_plot(dashboard_fig, "improved_system/advanced_dashboard.png")
        print("   ✓ Комплексный дашборд сохранен: improved_system/advanced_dashboard.png")
        
        # Отображение графика
        print("   Отображение комплексного дашборда...")
        plt.show()
        input("   Нажмите Enter для продолжения к следующему графику...")
        plt.close()
        
        # График с аномалиями
        print("   Создание графика обнаружения аномалий...")
        anomalies = analyzer.detect_anomalies()
        anomaly_fig = visualizer.create_anomaly_plot(
            metrics=analyzer.analysis_data.performance_metrics,
            anomalies=anomalies
        )
        
        visualizer.save_plot(anomaly_fig, "improved_system/anomaly_detection.png")
        print("   ✓ График обнаружения аномалий сохранен: improved_system/anomaly_detection.png")
        
        # Отображение графика
        print("   Отображение графика обнаружения аномалий...")
        plt.show()
        input("   Нажмите Enter для продолжения к следующему графику...")
        plt.close()
        
        # График с прогнозами
        print("   Создание графика прогнозирования отказов...")
        predictions = analyzer.predict_failures()
        prediction_fig = visualizer.create_prediction_plot(
            metrics=analyzer.analysis_data.performance_metrics,
            predictions=predictions
        )
        
        visualizer.save_plot(prediction_fig, "improved_system/failure_predictions.png")
        print("   ✓ График прогнозирования отказов сохранен: improved_system/failure_predictions.png")
        
        # Отображение графика
        print("   Отображение графика прогнозирования отказов...")
        plt.show()
        input("   Нажмите Enter для завершения...")
        plt.close()
        
    except Exception as e:
        print(f"   ✗ Ошибка создания визуализаций: {e}")


def main():
    """
    Основная функция демонстрации улучшенной системы анализа
    """
    print("="*80)
    print("УЛУЧШЕННАЯ СИСТЕМА АНАЛИЗА ФУНКЦИОНИРОВАНИЯ ИКС В НЕБЛАГОПРИЯТНЫХ УСЛОВИЯХ")
    print("Версия 2.0 - с поддержкой машинного обучения и расширенной аналитики")
    print("="*80)
    
    try:
        # Создание анализатора с тестовыми данными
        analyzer = create_realistic_sample_data()
        
        # Демонстрация расширенных возможностей
        demonstrate_advanced_analysis(analyzer)
        
        # Создание визуализаций с отображением
        create_advanced_visualizations(analyzer)
        
        # Генерация расширенного отчета
        print("\n9. Генерация расширенного отчета...")
        report = analyzer.generate_comprehensive_report()
        
        # Сохранение отчета
        with open("improved_system/advanced_analysis_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        print("   ✓ Расширенный отчет сохранен: improved_system/advanced_analysis_report.txt")
        
        # Сохранение данных анализа
        analyzer.save_analysis_data("improved_system/advanced_analysis_data.json")
        print("   ✓ Данные анализа сохранены: improved_system/advanced_analysis_data.json")
        
        # Экспорт в Excel
        try:
            analyzer.export_to_excel("improved_system/advanced_analysis_data.xlsx")
            print("   ✓ Данные экспортированы в Excel: improved_system/advanced_analysis_data.xlsx")
        except ImportError:
            print("   ! openpyxl не установлен, экспорт в Excel недоступен")
        
        print("\n" + "="*80)
        print("АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
        print("="*80)
        print("\nСозданные файлы:")
        print("  - improved_system/advanced_analysis_report.txt - расширенный отчет")
        print("  - improved_system/advanced_analysis_data.json - данные в JSON")
        print("  - improved_system/advanced_analysis_data.xlsx - данные в Excel (если доступно)")
        print("  - improved_system/advanced_dashboard.png - комплексный дашборд")
        print("  - improved_system/anomaly_detection.png - обнаружение аномалий")
        print("  - improved_system/failure_predictions.png - прогнозирование отказов")
        print("  - improved_system/improved_system_analysis.log - лог работы системы")
        print("  - improved_system/config.yaml - конфигурация системы")
        
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        print(f"\n❌ Критическая ошибка: {e}")
        print("Проверьте логи в файле improved_system/improved_system_analysis.log")


if __name__ == "__main__":
    main()
