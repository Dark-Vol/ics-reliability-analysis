#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ функционирования информационно-коммуникационных систем в неблагоприятных условиях

Этот модуль содержит основные классы и функции для анализа работы
информационно-коммуникационных систем в экстремальных условиях эксплуатации.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SystemAnalyzer:
    """
    Класс для анализа функционирования информационно-коммуникационных систем
    в неблагоприятных условиях.
    """
    
    def __init__(self, system_name: str):
        """
        Инициализация анализатора системы.
        
        Args:
            system_name (str): Название анализируемой системы
        """
        self.system_name = system_name
        self.performance_data = {}
        self.failure_events = []
        self.environmental_conditions = {}
        
    def add_performance_metric(self, metric_name: str, values: List[float], 
                             timestamps: List[datetime]) -> None:
        """
        Добавить метрику производительности системы.
        
        Args:
            metric_name (str): Название метрики
            values (List[float]): Значения метрики
            timestamps (List[datetime]): Временные метки
        """
        self.performance_data[metric_name] = {
            'values': values,
            'timestamps': timestamps
        }
        logger.info(f"Добавлена метрика '{metric_name}' с {len(values)} значениями")
    
    def add_failure_event(self, timestamp: datetime, event_type: str, 
                         description: str, severity: str) -> None:
        """
        Добавить событие отказа системы.
        
        Args:
            timestamp (datetime): Время события
            event_type (str): Тип события
            description (str): Описание события
            severity (str): Уровень критичности
        """
        event = {
            'timestamp': timestamp,
            'event_type': event_type,
            'description': description,
            'severity': severity
        }
        self.failure_events.append(event)
        logger.warning(f"Зарегистрирован отказ: {event_type} - {description}")
    
    def add_environmental_condition(self, condition_name: str, 
                                  values: List[float], 
                                  timestamps: List[datetime]) -> None:
        """
        Добавить данные о внешних условиях.
        
        Args:
            condition_name (str): Название условия
            values (List[float]): Значения условия
            timestamps (List[datetime]): Временные метки
        """
        self.environmental_conditions[condition_name] = {
            'values': values,
            'timestamps': timestamps
        }
        logger.info(f"Добавлено условие '{condition_name}' с {len(values)} значениями")
    
    def calculate_reliability_metrics(self) -> Dict[str, float]:
        """
        Расчет показателей надежности системы.
        
        Returns:
            Dict[str, float]: Словарь с метриками надежности
        """
        if not self.failure_events:
            return {'MTBF': float('inf'), 'MTTR': 0.0, 'Availability': 1.0}
        
        # Сортировка событий по времени
        sorted_events = sorted(self.failure_events, key=lambda x: x['timestamp'])
        
        # Расчет MTBF (Mean Time Between Failures)
        if len(sorted_events) > 1:
            time_diffs = []
            for i in range(1, len(sorted_events)):
                diff = (sorted_events[i]['timestamp'] - 
                       sorted_events[i-1]['timestamp']).total_seconds() / 3600  # в часах
                time_diffs.append(diff)
            mtbf = np.mean(time_diffs) if time_diffs else float('inf')
        else:
            mtbf = float('inf')
        
        # Расчет MTTR (Mean Time To Repair) - упрощенная модель
        mttr = 2.0  # Предполагаемое время восстановления в часах
        
        # Расчет доступности
        total_time = 24 * 30  # 30 дней в часах (пример)
        downtime = len(self.failure_events) * mttr
        availability = max(0, (total_time - downtime) / total_time)
        
        metrics = {
            'MTBF': mtbf,
            'MTTR': mttr,
            'Availability': availability,
            'Total_Failures': len(self.failure_events)
        }
        
        logger.info(f"Рассчитаны метрики надежности: {metrics}")
        return metrics
    
    def analyze_performance_degradation(self) -> Dict[str, any]:
        """
        Анализ деградации производительности системы.
        
        Returns:
            Dict[str, any]: Результаты анализа деградации
        """
        degradation_analysis = {}
        
        for metric_name, data in self.performance_data.items():
            values = np.array(data['values'])
            
            # Расчет тренда
            x = np.arange(len(values))
            trend = np.polyfit(x, values, 1)[0]
            
            # Расчет коэффициента вариации
            cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
            
            # Определение критических точек
            threshold = np.mean(values) - 2 * np.std(values)
            critical_points = np.sum(values < threshold)
            
            degradation_analysis[metric_name] = {
                'trend': trend,
                'coefficient_of_variation': cv,
                'critical_points': critical_points,
                'mean_value': np.mean(values),
                'std_value': np.std(values)
            }
        
        logger.info("Проведен анализ деградации производительности")
        return degradation_analysis
    
    def correlate_with_environment(self) -> Dict[str, float]:
        """
        Корреляционный анализ между производительностью и внешними условиями.
        
        Returns:
            Dict[str, float]: Словарь с коэффициентами корреляции
        """
        correlations = {}
        
        for metric_name, metric_data in self.performance_data.items():
            metric_values = np.array(metric_data['values'])
            
            for condition_name, condition_data in self.environmental_conditions.items():
                condition_values = np.array(condition_data['values'])
                
                # Приведение к одинаковой длине
                min_length = min(len(metric_values), len(condition_values))
                if min_length > 1:
                    correlation = np.corrcoef(
                        metric_values[:min_length], 
                        condition_values[:min_length]
                    )[0, 1]
                    
                    if not np.isnan(correlation):
                        key = f"{metric_name}_vs_{condition_name}"
                        correlations[key] = correlation
        
        logger.info(f"Рассчитаны корреляции: {len(correlations)} пар")
        return correlations
    
    def generate_report(self) -> str:
        """
        Генерация отчета о функционировании системы.
        
        Returns:
            str: Текстовый отчет
        """
        report = f"""
=== ОТЧЕТ О ФУНКЦИОНИРОВАНИИ СИСТЕМЫ ===
Система: {self.system_name}
Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. МЕТРИКИ НАДЕЖНОСТИ:
"""
        
        reliability_metrics = self.calculate_reliability_metrics()
        for metric, value in reliability_metrics.items():
            report += f"   {metric}: {value:.4f}\n"
        
        report += "\n2. АНАЛИЗ ДЕГРАДАЦИИ ПРОИЗВОДИТЕЛЬНОСТИ:\n"
        degradation = self.analyze_performance_degradation()
        for metric, data in degradation.items():
            report += f"   {metric}:\n"
            report += f"     Тренд: {data['trend']:.6f}\n"
            report += f"     Коэффициент вариации: {data['coefficient_of_variation']:.4f}\n"
            report += f"     Критические точки: {data['critical_points']}\n"
        
        report += "\n3. КОРРЕЛЯЦИИ С ВНЕШНИМИ УСЛОВИЯМИ:\n"
        correlations = self.correlate_with_environment()
        for pair, corr in correlations.items():
            report += f"   {pair}: {corr:.4f}\n"
        
        report += f"\n4. СОБЫТИЯ ОТКАЗОВ ({len(self.failure_events)}):\n"
        for i, event in enumerate(self.failure_events[-5:], 1):  # Последние 5 событий
            report += f"   {i}. {event['timestamp']} - {event['event_type']}: {event['description']}\n"
        
        return report
    
    def save_data_to_json(self, filename: str) -> None:
        """
        Сохранение данных в JSON файл.
        
        Args:
            filename (str): Имя файла для сохранения
        """
        data = {
            'system_name': self.system_name,
            'performance_data': {
                name: {
                    'values': values['values'],
                    'timestamps': [ts.isoformat() for ts in values['timestamps']]
                }
                for name, values in self.performance_data.items()
            },
            'failure_events': [
                {
                    'timestamp': event['timestamp'].isoformat(),
                    'event_type': event['event_type'],
                    'description': event['description'],
                    'severity': event['severity']
                }
                for event in self.failure_events
            ],
            'environmental_conditions': {
                name: {
                    'values': values['values'],
                    'timestamps': [ts.isoformat() for ts in values['timestamps']]
                }
                for name, values in self.environmental_conditions.items()
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Данные сохранены в файл: {filename}")


def create_sample_data() -> SystemAnalyzer:
    """
    Создание примера данных для демонстрации работы системы.
    
    Returns:
        SystemAnalyzer: Настроенный анализатор с тестовыми данными
    """
    analyzer = SystemAnalyzer("Тестовая ИКС")
    
    # Генерация тестовых данных
    np.random.seed(42)
    start_time = datetime.now() - timedelta(days=30)
    
    # Данные о производительности
    timestamps = [start_time + timedelta(hours=i) for i in range(720)]  # 30 дней по часам
    
    # CPU загрузка с трендом деградации
    cpu_values = 50 + 20 * np.sin(np.linspace(0, 4*np.pi, 720)) + np.random.normal(0, 5, 720)
    cpu_values += np.linspace(0, 10, 720)  # Тренд деградации
    
    # Пропускная способность
    throughput_values = 1000 + 200 * np.sin(np.linspace(0, 2*np.pi, 720)) + np.random.normal(0, 50, 720)
    throughput_values -= np.linspace(0, 100, 720)  # Тренд деградации
    
    # Внешние условия
    temperature_values = 20 + 10 * np.sin(np.linspace(0, 2*np.pi, 720)) + np.random.normal(0, 2, 720)
    humidity_values = 60 + 20 * np.sin(np.linspace(0, 3*np.pi, 720)) + np.random.normal(0, 5, 720)
    
    # Добавление данных в анализатор
    analyzer.add_performance_metric("CPU_Usage", cpu_values.tolist(), timestamps)
    analyzer.add_performance_metric("Throughput", throughput_values.tolist(), timestamps)
    analyzer.add_environmental_condition("Temperature", temperature_values.tolist(), timestamps)
    analyzer.add_environmental_condition("Humidity", humidity_values.tolist(), timestamps)
    
    # Добавление событий отказов
    failure_times = [start_time + timedelta(days=i*7) for i in range(1, 5)]
    failure_types = ["Hardware", "Software", "Network", "Power"]
    descriptions = [
        "Отказ процессора из-за перегрева",
        "Ошибка в модуле обработки данных", 
        "Потеря связи с удаленным узлом",
        "Сбой питания в критическом узле"
    ]
    
    for i, (time, ftype, desc) in enumerate(zip(failure_times, failure_types, descriptions)):
        analyzer.add_failure_event(time, ftype, desc, "High" if i % 2 == 0 else "Medium")
    
    return analyzer


def main():
    """
    Основная функция для демонстрации работы системы анализа.
    """
    print("=== АНАЛИЗ ФУНКЦИОНИРОВАНИЯ ИКС В НЕБЛАГОПРИЯТНЫХ УСЛОВИЯХ ===\n")
    
    # Создание анализатора с тестовыми данными
    analyzer = create_sample_data()
    
    # Генерация и вывод отчета
    report = analyzer.generate_report()
    print(report)
    
    # Сохранение данных
    analyzer.save_data_to_json("system_analysis_data.json")
    
    # Создание простого графика
    try:
        plt.figure(figsize=(12, 8))
        
        # График производительности
        plt.subplot(2, 2, 1)
        cpu_data = analyzer.performance_data["CPU_Usage"]
        plt.plot(cpu_data['timestamps'][::24], cpu_data['values'][::24])  # Каждый день
        plt.title("Загрузка CPU")
        plt.ylabel("Процент")
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        throughput_data = analyzer.performance_data["Throughput"]
        plt.plot(throughput_data['timestamps'][::24], throughput_data['values'][::24])
        plt.title("Пропускная способность")
        plt.ylabel("Мбит/с")
        plt.xticks(rotation=45)
        
        # График внешних условий
        plt.subplot(2, 2, 3)
        temp_data = analyzer.environmental_conditions["Temperature"]
        plt.plot(temp_data['timestamps'][::24], temp_data['values'][::24])
        plt.title("Температура")
        plt.ylabel("°C")
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 4)
        humidity_data = analyzer.environmental_conditions["Humidity"]
        plt.plot(humidity_data['timestamps'][::24], humidity_data['values'][::24])
        plt.title("Влажность")
        plt.ylabel("Процент")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig("system_analysis_plot.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nГрафик сохранен как 'system_analysis_plot.png'")
        
    except ImportError:
        print("Matplotlib не установлен. График не создан.")
    
    print("\nАнализ завершен. Данные сохранены в 'system_analysis_data.json'")


if __name__ == "__main__":
    main()
