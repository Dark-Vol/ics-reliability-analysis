"""
Модели данных для системы анализа ИКС
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import json


class SeverityLevel(Enum):
    """Уровни критичности событий"""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class EventType(Enum):
    """Типы событий системы"""
    HARDWARE = "Hardware"
    SOFTWARE = "Software"
    NETWORK = "Network"
    POWER = "Power"
    ENVIRONMENTAL = "Environmental"
    HUMAN = "Human"
    SECURITY = "Security"


@dataclass
class PerformanceMetric:
    """Метрика производительности системы"""
    name: str
    values: List[float]
    timestamps: List[datetime]
    unit: str = ""
    description: str = ""
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    
    def __post_init__(self):
        """Валидация данных после инициализации"""
        if len(self.values) != len(self.timestamps):
            raise ValueError("Количество значений должно соответствовать количеству временных меток")
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для сериализации"""
        return {
            'name': self.name,
            'values': self.values,
            'timestamps': [ts.isoformat() for ts in self.timestamps],
            'unit': self.unit,
            'description': self.description,
            'threshold_warning': self.threshold_warning,
            'threshold_critical': self.threshold_critical
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetric':
        """Создание из словаря"""
        return cls(
            name=data['name'],
            values=data['values'],
            timestamps=[datetime.fromisoformat(ts) for ts in data['timestamps']],
            unit=data.get('unit', ''),
            description=data.get('description', ''),
            threshold_warning=data.get('threshold_warning'),
            threshold_critical=data.get('threshold_critical')
        )


@dataclass
class FailureEvent:
    """Событие отказа системы"""
    timestamp: datetime
    event_type: EventType
    description: str
    severity: SeverityLevel
    duration: Optional[float] = None  # в часах
    recovery_time: Optional[datetime] = None
    affected_components: List[str] = field(default_factory=list)
    root_cause: Optional[str] = None
    resolution: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для сериализации"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'description': self.description,
            'severity': self.severity.value,
            'duration': self.duration,
            'recovery_time': self.recovery_time.isoformat() if self.recovery_time else None,
            'affected_components': self.affected_components,
            'root_cause': self.root_cause,
            'resolution': self.resolution
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FailureEvent':
        """Создание из словаря"""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            event_type=EventType(data['event_type']),
            description=data['description'],
            severity=SeverityLevel(data['severity']),
            duration=data.get('duration'),
            recovery_time=datetime.fromisoformat(data['recovery_time']) if data.get('recovery_time') else None,
            affected_components=data.get('affected_components', []),
            root_cause=data.get('root_cause'),
            resolution=data.get('resolution')
        )


@dataclass
class EnvironmentalCondition:
    """Внешнее условие окружающей среды"""
    name: str
    values: List[float]
    timestamps: List[datetime]
    unit: str = ""
    description: str = ""
    normal_range: Optional[tuple] = None  # (min, max)
    critical_range: Optional[tuple] = None  # (min, max)
    
    def __post_init__(self):
        """Валидация данных после инициализации"""
        if len(self.values) != len(self.timestamps):
            raise ValueError("Количество значений должно соответствовать количеству временных меток")
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для сериализации"""
        return {
            'name': self.name,
            'values': self.values,
            'timestamps': [ts.isoformat() for ts in self.timestamps],
            'unit': self.unit,
            'description': self.description,
            'normal_range': self.normal_range,
            'critical_range': self.critical_range
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnvironmentalCondition':
        """Создание из словаря"""
        return cls(
            name=data['name'],
            values=data['values'],
            timestamps=[datetime.fromisoformat(ts) for ts in data['timestamps']],
            unit=data.get('unit', ''),
            description=data.get('description', ''),
            normal_range=data.get('normal_range'),
            critical_range=data.get('critical_range')
        )


@dataclass
class ReliabilityMetrics:
    """Метрики надежности системы"""
    mtbf: float  # Mean Time Between Failures (часы)
    mttr: float  # Mean Time To Repair (часы)
    mttf: float  # Mean Time To Failure (часы)
    availability: float  # Доступность (0-1)
    rto: float  # Recovery Time Objective (часы)
    rpo: float  # Recovery Point Objective (часы)
    total_failures: int
    critical_failures: int
    uptime: float  # Время работы (часы)
    downtime: float  # Время простоя (часы)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'mtbf': self.mtbf,
            'mttr': self.mttr,
            'mttf': self.mttf,
            'availability': self.availability,
            'rto': self.rto,
            'rpo': self.rpo,
            'total_failures': self.total_failures,
            'critical_failures': self.critical_failures,
            'uptime': self.uptime,
            'downtime': self.downtime
        }


@dataclass
class DegradationAnalysis:
    """Анализ деградации производительности"""
    metric_name: str
    trend_slope: float
    trend_p_value: float
    coefficient_of_variation: float
    critical_points: int
    mean_value: float
    std_value: float
    min_value: float
    max_value: float
    degradation_rate: float  # % в час
    is_degrading: bool
    confidence_level: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'metric_name': self.metric_name,
            'trend_slope': self.trend_slope,
            'trend_p_value': self.trend_p_value,
            'coefficient_of_variation': self.coefficient_of_variation,
            'critical_points': self.critical_points,
            'mean_value': self.mean_value,
            'std_value': self.std_value,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'degradation_rate': self.degradation_rate,
            'is_degrading': self.is_degrading,
            'confidence_level': self.confidence_level
        }


@dataclass
class CorrelationResult:
    """Результат корреляционного анализа"""
    metric_name: str
    condition_name: str
    correlation_coefficient: float
    p_value: float
    significance_level: float
    is_significant: bool
    relationship_strength: str  # weak, moderate, strong
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'metric_name': self.metric_name,
            'condition_name': self.condition_name,
            'correlation_coefficient': self.correlation_coefficient,
            'p_value': self.p_value,
            'significance_level': self.significance_level,
            'is_significant': self.is_significant,
            'relationship_strength': self.relationship_strength
        }


@dataclass
class SystemAnalysisData:
    """Основной контейнер данных для анализа системы"""
    system_name: str
    analysis_timestamp: datetime
    performance_metrics: List[PerformanceMetric] = field(default_factory=list)
    failure_events: List[FailureEvent] = field(default_factory=list)
    environmental_conditions: List[EnvironmentalCondition] = field(default_factory=list)
    reliability_metrics: Optional[ReliabilityMetrics] = None
    degradation_analysis: List[DegradationAnalysis] = field(default_factory=list)
    correlation_results: List[CorrelationResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для сериализации"""
        return {
            'system_name': self.system_name,
            'analysis_timestamp': self.analysis_timestamp.isoformat(),
            'performance_metrics': [m.to_dict() for m in self.performance_metrics],
            'failure_events': [e.to_dict() for e in self.failure_events],
            'environmental_conditions': [c.to_dict() for c in self.environmental_conditions],
            'reliability_metrics': self.reliability_metrics.to_dict() if self.reliability_metrics else None,
            'degradation_analysis': [d.to_dict() for d in self.degradation_analysis],
            'correlation_results': [c.to_dict() for c in self.correlation_results]
        }
    
    def save_to_json(self, filename: str) -> None:
        """Сохранение в JSON файл"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_from_json(cls, filename: str) -> 'SystemAnalysisData':
        """Загрузка из JSON файла"""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Создание объекта с базовыми данными
        obj = cls(
            system_name=data['system_name'],
            analysis_timestamp=datetime.fromisoformat(data['analysis_timestamp'])
        )
        
        # Загрузка метрик производительности
        for metric_data in data.get('performance_metrics', []):
            obj.performance_metrics.append(PerformanceMetric.from_dict(metric_data))
        
        # Загрузка событий отказов
        for event_data in data.get('failure_events', []):
            obj.failure_events.append(FailureEvent.from_dict(event_data))
        
        # Загрузка внешних условий
        for condition_data in data.get('environmental_conditions', []):
            obj.environmental_conditions.append(EnvironmentalCondition.from_dict(condition_data))
        
        # Загрузка метрик надежности
        if data.get('reliability_metrics'):
            obj.reliability_metrics = ReliabilityMetrics(**data['reliability_metrics'])
        
        return obj
