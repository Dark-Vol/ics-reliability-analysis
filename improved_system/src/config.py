"""
Конфигурационный модуль для системы анализа ИКС
"""

import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Config:
    """Класс для управления конфигурацией системы анализа"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Инициализация конфигурации
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        self.config_path = config_path or "config.yaml"
        self._config = self._load_default_config()
        
        if Path(self.config_path).exists():
            self._load_config()
        else:
            self._save_default_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Загрузка конфигурации по умолчанию"""
        return {
            'system': {
                'name': 'ИКС Анализ',
                'version': '2.0.0',
                'log_level': 'INFO',
                'log_file': 'system_analysis.log'
            },
            'reliability': {
                'mtbf_threshold': 168.0,  # часов
                'mttr_threshold': 4.0,    # часов
                'availability_threshold': 0.99,
                'failure_severity_weights': {
                    'Critical': 1.0,
                    'High': 0.8,
                    'Medium': 0.6,
                    'Low': 0.4
                }
            },
            'analysis': {
                'trend_window': 24,  # часов
                'correlation_threshold': 0.7,
                'anomaly_threshold': 3.0,  # стандартных отклонений
                'degradation_threshold': -0.1  # наклон тренда
            },
            'ml': {
                'enabled': True,
                'models': {
                    'failure_prediction': {
                        'type': 'RandomForest',
                        'n_estimators': 100,
                        'max_depth': 10
                    },
                    'anomaly_detection': {
                        'type': 'IsolationForest',
                        'contamination': 0.1
                    },
                    'time_series': {
                        'type': 'LSTM',
                        'sequence_length': 24,
                        'epochs': 50
                    }
                }
            },
            'visualization': {
                'figure_size': [12, 8],
                'dpi': 300,
                'style': 'seaborn-v0_8',
                'colors': {
                    'primary': '#1f77b4',
                    'secondary': '#ff7f0e',
                    'success': '#2ca02c',
                    'warning': '#d62728',
                    'info': '#9467bd'
                }
            },
            'data': {
                'input_formats': ['json', 'csv', 'excel'],
                'output_formats': ['json', 'csv', 'excel', 'pdf'],
                'backup_enabled': True,
                'compression': True
            }
        }
    
    def _load_config(self) -> None:
        """Загрузка конфигурации из файла"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
                self._merge_config(file_config)
            logger.info(f"Конфигурация загружена из {self.config_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
    
    def _save_default_config(self) -> None:
        """Сохранение конфигурации по умолчанию"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            logger.info(f"Конфигурация по умолчанию сохранена в {self.config_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения конфигурации: {e}")
    
    def _merge_config(self, new_config: Dict[str, Any]) -> None:
        """Рекурсивное слияние конфигураций"""
        def merge_dicts(d1: Dict, d2: Dict) -> Dict:
            for key, value in d2.items():
                if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
                    merge_dicts(d1[key], value)
                else:
                    d1[key] = value
            return d1
        
        self._config = merge_dicts(self._config, new_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Получение значения конфигурации по ключу
        
        Args:
            key: Ключ конфигурации (например, 'reliability.mtbf_threshold')
            default: Значение по умолчанию
            
        Returns:
            Значение конфигурации
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Установка значения конфигурации
        
        Args:
            key: Ключ конфигурации
            value: Новое значение
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.info(f"Конфигурация обновлена: {key} = {value}")
    
    def save(self) -> None:
        """Сохранение конфигурации в файл"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            logger.info("Конфигурация сохранена")
        except Exception as e:
            logger.error(f"Ошибка сохранения конфигурации: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Возврат конфигурации как словарь"""
        return self._config.copy()
