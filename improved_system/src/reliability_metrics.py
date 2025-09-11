"""
Модуль для расчета метрик надежности ИКС
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from .data_models import FailureEvent, ReliabilityMetrics, SeverityLevel

logger = logging.getLogger(__name__)


class ReliabilityCalculator:
    """Калькулятор метрик надежности системы"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация калькулятора
        
        Args:
            config: Конфигурация с пороговыми значениями
        """
        self.config = config or {}
        self.severity_weights = self.config.get('failure_severity_weights', {
            'Critical': 1.0,
            'High': 0.8,
            'Medium': 0.6,
            'Low': 0.4
        })
    
    def calculate_metrics(self, failure_events: List[FailureEvent], 
                         analysis_period_hours: float = 720) -> ReliabilityMetrics:
        """
        Расчет всех метрик надежности
        
        Args:
            failure_events: Список событий отказов
            analysis_period_hours: Период анализа в часах
            
        Returns:
            Объект с метриками надежности
        """
        if not failure_events:
            return self._get_perfect_reliability(analysis_period_hours)
        
        # Сортировка событий по времени
        sorted_events = sorted(failure_events, key=lambda x: x.timestamp)
        
        # Расчет основных метрик
        mtbf = self._calculate_mtbf(sorted_events)
        mttr = self._calculate_mttr(sorted_events)
        mttf = self._calculate_mttf(sorted_events, analysis_period_hours)
        availability = self._calculate_availability(sorted_events, analysis_period_hours)
        rto, rpo = self._calculate_rto_rpo(sorted_events)
        
        # Подсчет отказов по критичности
        total_failures = len(failure_events)
        critical_failures = len([e for e in failure_events 
                               if e.severity == SeverityLevel.CRITICAL])
        
        # Расчет времени работы и простоя
        uptime, downtime = self._calculate_uptime_downtime(sorted_events, analysis_period_hours)
        
        return ReliabilityMetrics(
            mtbf=mtbf,
            mttr=mttr,
            mttf=mttf,
            availability=availability,
            rto=rto,
            rpo=rpo,
            total_failures=total_failures,
            critical_failures=critical_failures,
            uptime=uptime,
            downtime=downtime
        )
    
    def _get_perfect_reliability(self, analysis_period_hours: float) -> ReliabilityMetrics:
        """Возврат идеальных метрик для системы без отказов"""
        return ReliabilityMetrics(
            mtbf=float('inf'),
            mttr=0.0,
            mttf=analysis_period_hours,
            availability=1.0,
            rto=0.0,
            rpo=0.0,
            total_failures=0,
            critical_failures=0,
            uptime=analysis_period_hours,
            downtime=0.0
        )
    
    def _calculate_mtbf(self, sorted_events: List[FailureEvent]) -> float:
        """
        Расчет MTBF (Mean Time Between Failures)
        
        Args:
            sorted_events: Отсортированные по времени события отказов
            
        Returns:
            MTBF в часах
        """
        if len(sorted_events) < 2:
            return float('inf')
        
        # Расчет интервалов между отказами
        time_diffs = []
        for i in range(1, len(sorted_events)):
            diff_hours = (sorted_events[i].timestamp - 
                         sorted_events[i-1].timestamp).total_seconds() / 3600
            time_diffs.append(diff_hours)
        
        # Взвешенное среднее с учетом критичности
        weighted_diffs = []
        for i, diff in enumerate(time_diffs):
            weight = self.severity_weights.get(sorted_events[i].severity.value, 1.0)
            weighted_diffs.append(diff / weight)
        
        return np.mean(weighted_diffs) if weighted_diffs else float('inf')
    
    def _calculate_mttr(self, sorted_events: List[FailureEvent]) -> float:
        """
        Расчет MTTR (Mean Time To Repair)
        
        Args:
            sorted_events: Отсортированные по времени события отказов
            
        Returns:
            MTTR в часах
        """
        repair_times = []
        
        for event in sorted_events:
            if event.duration is not None:
                # Используем реальное время восстановления
                repair_times.append(event.duration)
            elif event.recovery_time is not None:
                # Вычисляем время восстановления
                duration = (event.recovery_time - event.timestamp).total_seconds() / 3600
                repair_times.append(duration)
            else:
                # Используем эмпирические значения в зависимости от типа и критичности
                base_time = self._get_base_repair_time(event.event_type, event.severity)
                repair_times.append(base_time)
        
        if not repair_times:
            return 2.0  # Значение по умолчанию
        
        # Взвешенное среднее с учетом критичности
        weighted_times = []
        for i, time in enumerate(repair_times):
            weight = self.severity_weights.get(sorted_events[i].severity.value, 1.0)
            weighted_times.append(time * weight)
        
        return np.mean(weighted_times)
    
    def _get_base_repair_time(self, event_type, severity: SeverityLevel) -> float:
        """
        Получение базового времени восстановления в зависимости от типа и критичности
        
        Args:
            event_type: Тип события
            severity: Критичность события
            
        Returns:
            Базовое время восстановления в часах
        """
        # Базовые времена восстановления по типам (в часах)
        base_times = {
            'Hardware': 4.0,
            'Software': 2.0,
            'Network': 1.0,
            'Power': 0.5,
            'Environmental': 6.0,
            'Human': 1.5,
            'Security': 8.0
        }
        
        base_time = base_times.get(event_type.value, 2.0)
        
        # Модификация в зависимости от критичности
        severity_multipliers = {
            SeverityLevel.CRITICAL: 1.5,
            SeverityLevel.HIGH: 1.2,
            SeverityLevel.MEDIUM: 1.0,
            SeverityLevel.LOW: 0.8
        }
        
        multiplier = severity_multipliers.get(severity, 1.0)
        return base_time * multiplier
    
    def _calculate_mttf(self, sorted_events: List[FailureEvent], 
                       analysis_period_hours: float) -> float:
        """
        Расчет MTTF (Mean Time To Failure)
        
        Args:
            sorted_events: Отсортированные по времени события отказов
            analysis_period_hours: Период анализа в часах
            
        Returns:
            MTTF в часах
        """
        if not sorted_events:
            return analysis_period_hours
        
        # Время до первого отказа
        first_failure_time = (sorted_events[0].timestamp - 
                             (sorted_events[0].timestamp - timedelta(hours=analysis_period_hours))).total_seconds() / 3600
        
        # Среднее время между отказами
        mtbf = self._calculate_mtbf(sorted_events)
        
        # MTTF = время до первого отказа + MTBF
        return first_failure_time + mtbf
    
    def _calculate_availability(self, sorted_events: List[FailureEvent], 
                              analysis_period_hours: float) -> float:
        """
        Расчет доступности системы
        
        Args:
            sorted_events: События отказов
            analysis_period_hours: Период анализа в часах
            
        Returns:
            Коэффициент доступности (0-1)
        """
        if not sorted_events:
            return 1.0
        
        # Расчет общего времени простоя
        total_downtime = 0.0
        for event in sorted_events:
            if event.duration is not None:
                total_downtime += event.duration
            elif event.recovery_time is not None:
                duration = (event.recovery_time - event.timestamp).total_seconds() / 3600
                total_downtime += duration
            else:
                # Используем базовое время восстановления
                base_time = self._get_base_repair_time(event.event_type, event.severity)
                total_downtime += base_time
        
        # Доступность = (общее время - время простоя) / общее время
        uptime = analysis_period_hours - total_downtime
        availability = max(0.0, uptime / analysis_period_hours)
        
        return availability
    
    def _calculate_rto_rpo(self, sorted_events: List[FailureEvent]) -> Tuple[float, float]:
        """
        Расчет RTO (Recovery Time Objective) и RPO (Recovery Point Objective)
        
        Args:
            sorted_events: События отказов
            
        Returns:
            Кортеж (RTO, RPO) в часах
        """
        if not sorted_events:
            return 0.0, 0.0
        
        # RTO - максимальное время восстановления
        repair_times = []
        for event in sorted_events:
            if event.duration is not None:
                repair_times.append(event.duration)
            elif event.recovery_time is not None:
                duration = (event.recovery_time - event.timestamp).total_seconds() / 3600
                repair_times.append(duration)
            else:
                base_time = self._get_base_repair_time(event.event_type, event.severity)
                repair_times.append(base_time)
        
        rto = max(repair_times) if repair_times else 0.0
        
        # RPO - максимальная потеря данных (упрощенная оценка)
        # В реальной системе это зависит от частоты резервного копирования
        rpo = rto * 0.1  # 10% от времени восстановления
        
        return rto, rpo
    
    def _calculate_uptime_downtime(self, sorted_events: List[FailureEvent], 
                                  analysis_period_hours: float) -> Tuple[float, float]:
        """
        Расчет времени работы и простоя
        
        Args:
            sorted_events: События отказов
            analysis_period_hours: Период анализа в часах
            
        Returns:
            Кортеж (время работы, время простоя) в часах
        """
        total_downtime = 0.0
        
        for event in sorted_events:
            if event.duration is not None:
                total_downtime += event.duration
            elif event.recovery_time is not None:
                duration = (event.recovery_time - event.timestamp).total_seconds() / 3600
                total_downtime += duration
            else:
                base_time = self._get_base_repair_time(event.event_type, event.severity)
                total_downtime += base_time
        
        uptime = max(0.0, analysis_period_hours - total_downtime)
        downtime = min(analysis_period_hours, total_downtime)
        
        return uptime, downtime
    
    def calculate_failure_rate(self, failure_events: List[FailureEvent], 
                              analysis_period_hours: float) -> float:
        """
        Расчет интенсивности отказов (failures per hour)
        
        Args:
            failure_events: События отказов
            analysis_period_hours: Период анализа в часах
            
        Returns:
            Интенсивность отказов (отказов в час)
        """
        if analysis_period_hours <= 0:
            return 0.0
        
        return len(failure_events) / analysis_period_hours
    
    def calculate_mtbf_confidence_interval(self, failure_events: List[FailureEvent], 
                                         confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Расчет доверительного интервала для MTBF
        
        Args:
            failure_events: События отказов
            confidence_level: Уровень доверия (0-1)
            
        Returns:
            Кортеж (нижняя граница, верхняя граница)
        """
        if len(failure_events) < 2:
            return float('inf'), float('inf')
        
        sorted_events = sorted(failure_events, key=lambda x: x.timestamp)
        time_diffs = []
        
        for i in range(1, len(sorted_events)):
            diff_hours = (sorted_events[i].timestamp - 
                         sorted_events[i-1].timestamp).total_seconds() / 3600
            time_diffs.append(diff_hours)
        
        if not time_diffs:
            return float('inf'), float('inf')
        
        mean_mtbf = np.mean(time_diffs)
        std_mtbf = np.std(time_diffs, ddof=1)
        n = len(time_diffs)
        
        # t-статистика для заданного уровня доверия
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence_level) / 2, n - 1)
        
        # Стандартная ошибка среднего
        se = std_mtbf / np.sqrt(n)
        
        # Доверительный интервал
        margin_error = t_value * se
        lower_bound = max(0, mean_mtbf - margin_error)
        upper_bound = mean_mtbf + margin_error
        
        return lower_bound, upper_bound
