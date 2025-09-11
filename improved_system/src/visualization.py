"""
Модуль для расширенной визуализации данных анализа ИКС
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Устанавливаем backend для отображения
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import warnings

from .data_models import (
    PerformanceMetric, FailureEvent, EnvironmentalCondition,
    ReliabilityMetrics, DegradationAnalysis, CorrelationResult
)

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


class AdvancedVisualizer:
    """Расширенный визуализатор для анализа ИКС"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация визуализатора
        
        Args:
            config: Конфигурация визуализации
        """
        self.config = config or {}
        
        # Настройки стиля
        self.style = self.config.get('style', 'seaborn-v0_8')
        self.figure_size = self.config.get('figure_size', [12, 8])
        self.dpi = self.config.get('dpi', 300)
        self.colors = self.config.get('colors', {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd'
        })
        
        # Настройка matplotlib
        plt.style.use(self.style)
        sns.set_palette("husl")
        
    def create_comprehensive_dashboard(self, metrics: List[PerformanceMetric],
                                     events: List[FailureEvent],
                                     conditions: List[EnvironmentalCondition],
                                     reliability: Optional[ReliabilityMetrics] = None,
                                     degradation: List[DegradationAnalysis] = None,
                                     correlations: List[CorrelationResult] = None) -> Figure:
        """
        Создание комплексного дашборда
        
        Args:
            metrics: Метрики производительности
            events: События отказов
            conditions: Внешние условия
            reliability: Метрики надежности
            degradation: Анализ деградации
            correlations: Результаты корреляций
            
        Returns:
            Объект Figure matplotlib
        """
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Комплексный анализ функционирования ИКС', fontsize=16, fontweight='bold')
        
        # Создание сетки графиков
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. График производительности (2x2)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_performance_metrics(ax1, metrics)
        
        # 2. График внешних условий (2x2)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_environmental_conditions(ax2, conditions)
        
        # 3. График событий отказов (1x4)
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_failure_events_timeline(ax3, events)
        
        # 4. Метрики надежности (1x2)
        ax4 = fig.add_subplot(gs[2, :2])
        self._plot_reliability_metrics(ax4, reliability)
        
        # 5. Анализ деградации (1x2)
        ax5 = fig.add_subplot(gs[2, 2:])
        self._plot_degradation_analysis(ax5, degradation)
        
        # 6. Корреляционная матрица (1x4)
        ax6 = fig.add_subplot(gs[3, :])
        self._plot_correlation_heatmap(ax6, correlations)
        
        return fig
    
    def _plot_performance_metrics(self, ax, metrics: List[PerformanceMetric]) -> None:
        """График метрик производительности"""
        ax.set_title('Метрики производительности', fontweight='bold')
        
        for i, metric in enumerate(metrics):
            if len(metric.values) > 0:
                # Уменьшение количества точек для лучшей читаемости
                step = max(1, len(metric.values) // 100)
                x_data = metric.timestamps[::step]
                y_data = metric.values[::step]
                
                ax.plot(x_data, y_data, label=metric.name, 
                       color=self.colors[f'primary' if i == 0 else f'secondary'],
                       linewidth=2, alpha=0.8)
                
                # Добавление пороговых значений
                if metric.threshold_warning:
                    ax.axhline(y=metric.threshold_warning, color='orange', 
                              linestyle='--', alpha=0.7, label=f'{metric.name} Warning')
                if metric.threshold_critical:
                    ax.axhline(y=metric.threshold_critical, color='red', 
                              linestyle='--', alpha=0.7, label=f'{metric.name} Critical')
        
        ax.set_ylabel('Значение')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Форматирование оси времени
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_environmental_conditions(self, ax, conditions: List[EnvironmentalCondition]) -> None:
        """График внешних условий"""
        ax.set_title('Внешние условия', fontweight='bold')
        
        for i, condition in enumerate(conditions):
            if len(condition.values) > 0:
                step = max(1, len(condition.values) // 100)
                x_data = condition.timestamps[::step]
                y_data = condition.values[::step]
                
                ax.plot(x_data, y_data, label=condition.name, 
                       color=self.colors[f'info' if i == 0 else f'warning'],
                       linewidth=2, alpha=0.8)
                
                # Добавление нормального диапазона
                if condition.normal_range:
                    ax.axhspan(condition.normal_range[0], condition.normal_range[1], 
                              alpha=0.2, color='green', label=f'{condition.name} Normal Range')
        
        ax.set_ylabel('Значение')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Форматирование оси времени
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_failure_events_timeline(self, ax, events: List[FailureEvent]) -> None:
        """Временная линия событий отказов"""
        ax.set_title('События отказов', fontweight='bold')
        
        if not events:
            ax.text(0.5, 0.5, 'Нет событий отказов', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            return
        
        # Группировка событий по типам
        event_types = {}
        for event in events:
            if event.event_type.value not in event_types:
                event_types[event.event_type.value] = []
            event_types[event.event_type.value].append(event)
        
        # Цвета для разных типов событий
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
        
        y_pos = 0
        for i, (event_type, type_events) in enumerate(event_types.items()):
            for event in type_events:
                # Высота маркера в зависимости от критичности
                height = {'Critical': 1.0, 'High': 0.8, 'Medium': 0.6, 'Low': 0.4}.get(
                    event.severity.value, 0.5)
                
                ax.scatter(event.timestamp, y_pos, 
                          s=100, c=colors[i % len(colors)], 
                          alpha=0.7, edgecolors='black', linewidth=1)
                
                # Добавление подписи для критических событий
                if event.severity.value == 'Critical':
                    ax.annotate(event.description[:20] + '...', 
                               (event.timestamp, y_pos),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
            
            y_pos += 1
        
        ax.set_ylabel('Тип события')
        ax.set_xlabel('Время')
        ax.grid(True, alpha=0.3)
        
        # Форматирование оси времени
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_reliability_metrics(self, ax, reliability: Optional[ReliabilityMetrics]) -> None:
        """График метрик надежности"""
        ax.set_title('Метрики надежности', fontweight='bold')
        
        if not reliability:
            ax.text(0.5, 0.5, 'Нет данных о надежности', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            return
        
        # Подготовка данных для графика
        metrics = ['MTBF', 'MTTR', 'MTTF', 'Availability', 'RTO', 'RPO']
        values = [reliability.mtbf, reliability.mttr, reliability.mttf, 
                 reliability.availability * 100, reliability.rto, reliability.rpo]
        
        # Нормализация значений для лучшего отображения
        normalized_values = []
        for i, (metric, value) in enumerate(zip(metrics, values)):
            if metric == 'Availability':
                normalized_values.append(value)  # Уже в процентах
            elif value == float('inf'):
                normalized_values.append(100)  # Максимальное значение
            else:
                # Нормализация к 0-100
                max_val = max([v for v in values if v != float('inf')])
                normalized_values.append(min(100, (value / max_val) * 100))
        
        # Создание столбчатой диаграммы
        bars = ax.bar(metrics, normalized_values, 
                     color=[self.colors['primary'], self.colors['secondary'], 
                           self.colors['success'], self.colors['info'],
                           self.colors['warning'], self.colors['primary']],
                     alpha=0.7, edgecolor='black', linewidth=1)
        
        # Добавление значений на столбцы
        for bar, value in zip(bars, values):
            if value != float('inf'):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       '∞', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Значение')
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_degradation_analysis(self, ax, degradation: List[DegradationAnalysis]) -> None:
        """График анализа деградации"""
        ax.set_title('Анализ деградации', fontweight='bold')
        
        if not degradation:
            ax.text(0.5, 0.5, 'Нет данных о деградации', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            return
        
        # Подготовка данных
        metric_names = [d.metric_name for d in degradation]
        degradation_rates = [d.degradation_rate for d in degradation]
        is_degrading = [d.is_degrading for d in degradation]
        
        # Цвета в зависимости от статуса деградации
        colors = [self.colors['warning'] if deg else self.colors['success'] 
                 for deg in is_degrading]
        
        # Создание горизонтальной столбчатой диаграммы
        bars = ax.barh(metric_names, degradation_rates, color=colors, alpha=0.7)
        
        # Добавление значений
        for bar, rate in zip(bars, degradation_rates):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{rate:.4f} %/ч', ha='left', va='center', fontweight='bold')
        
        ax.set_xlabel('Скорость деградации (%/час)')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='x')
    
    def _plot_correlation_heatmap(self, ax, correlations: List[CorrelationResult]) -> None:
        """Тепловая карта корреляций"""
        ax.set_title('Корреляционная матрица', fontweight='bold')
        
        if not correlations:
            ax.text(0.5, 0.5, 'Нет данных о корреляциях', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            return
        
        # Подготовка матрицы корреляций
        metric_names = list(set([c.metric_name for c in correlations]))
        condition_names = list(set([c.condition_name for c in correlations]))
        
        correlation_matrix = np.zeros((len(metric_names), len(condition_names)))
        
        for corr in correlations:
            i = metric_names.index(corr.metric_name)
            j = condition_names.index(corr.condition_name)
            correlation_matrix[i, j] = corr.correlation_coefficient
        
        # Создание тепловой карты
        im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        # Настройка осей
        ax.set_xticks(range(len(condition_names)))
        ax.set_yticks(range(len(metric_names)))
        ax.set_xticklabels(condition_names, rotation=45, ha='right')
        ax.set_yticklabels(metric_names)
        
        # Добавление значений в ячейки
        for i in range(len(metric_names)):
            for j in range(len(condition_names)):
                value = correlation_matrix[i, j]
                text_color = 'white' if abs(value) > 0.5 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                       color=text_color, fontweight='bold')
        
        # Добавление цветовой шкалы
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Коэффициент корреляции', rotation=270, labelpad=20)
    
    def create_anomaly_plot(self, metrics: List[PerformanceMetric], 
                           anomalies: List[Any]) -> Figure:
        """
        Создание графика с выделенными аномалиями
        
        Args:
            metrics: Метрики производительности
            anomalies: Результаты обнаружения аномалий
            
        Returns:
            Объект Figure matplotlib
        """
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        
        fig.suptitle('Обнаружение аномалий в метриках производительности', 
                    fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Основной график
            ax.plot(metric.timestamps, metric.values, 
                   color=self.colors['primary'], linewidth=2, alpha=0.7, label=metric.name)
            
            # Выделение аномалий
            if anomalies:
                anomaly_times = [a.timestamp for a in anomalies if a.is_anomaly]
                anomaly_values = []
                
                for time in anomaly_times:
                    # Поиск ближайшего значения метрики
                    time_diffs = [abs((ts - time).total_seconds()) for ts in metric.timestamps]
                    if time_diffs:
                        closest_idx = np.argmin(time_diffs)
                        if time_diffs[closest_idx] < 3600:  # В пределах часа
                            anomaly_values.append(metric.values[closest_idx])
                
                if anomaly_values:
                    ax.scatter(anomaly_times, anomaly_values, 
                              color=self.colors['warning'], s=100, 
                              alpha=0.8, edgecolors='black', linewidth=2,
                              label='Аномалии', zorder=5)
            
            # Добавление пороговых значений
            if metric.threshold_warning:
                ax.axhline(y=metric.threshold_warning, color='orange', 
                          linestyle='--', alpha=0.7, label='Warning')
            if metric.threshold_critical:
                ax.axhline(y=metric.threshold_critical, color='red', 
                          linestyle='--', alpha=0.7, label='Critical')
            
            ax.set_title(f'{metric.name} - Обнаружение аномалий')
            ax.set_ylabel(f'{metric.name} ({metric.unit})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Форматирование оси времени
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig
    
    def create_prediction_plot(self, metrics: List[PerformanceMetric], 
                              predictions: List[Any]) -> Figure:
        """
        Создание графика с прогнозами
        
        Args:
            metrics: Метрики производительности
            predictions: Результаты прогнозирования
            
        Returns:
            Объект Figure matplotlib
        """
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        
        fig.suptitle('Прогнозирование отказов системы', 
                    fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Основной график
            ax.plot(metric.timestamps, metric.values, 
                   color=self.colors['primary'], linewidth=2, alpha=0.7, label=metric.name)
            
            # Выделение прогнозов отказов
            if predictions:
                prediction_times = [p.timestamp for p in predictions if p.failure_probability > 0.1]
                prediction_probs = [p.failure_probability for p in predictions if p.failure_probability > 0.1]
                
                if prediction_times:
                    # Создание второго y-оси для вероятностей
                    ax2 = ax.twinx()
                    ax2.scatter(prediction_times, prediction_probs, 
                               color=self.colors['warning'], s=100, 
                               alpha=0.8, edgecolors='black', linewidth=2,
                               label='Вероятность отказа', zorder=5)
                    ax2.set_ylabel('Вероятность отказа', color=self.colors['warning'])
                    ax2.set_ylim(0, 1)
            
            ax.set_title(f'{metric.name} - Прогнозирование отказов')
            ax.set_ylabel(f'{metric.name} ({metric.unit})')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Форматирование оси времени
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig
    
    def save_plot(self, fig: Figure, filename: str, format: str = 'png') -> None:
        """
        Сохранение графика в файл
        
        Args:
            fig: Объект Figure matplotlib
            filename: Имя файла
            format: Формат файла ('png', 'pdf', 'svg')
        """
        try:
            fig.savefig(filename, format=format, dpi=self.dpi, 
                       bbox_inches='tight', facecolor='white', edgecolor='none')
            logger.info(f"График сохранен: {filename}")
        except Exception as e:
            logger.error(f"Ошибка сохранения графика: {e}")
