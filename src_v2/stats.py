from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import pandas as pd

@dataclass
class ImageStats:
    total_images: int
    basic_stats: Dict[str, float]
    quality_issues: Dict[str, List[str]]
    duplicates: Dict[str, Dict]
    quality_metrics: Dict[str, Dict]

class ReportGenerator:
    @staticmethod
    def generate_markdown(stats: ImageStats, output_path: Path) -> None:
        """Генерация отчета в формате Markdown"""
        report = [
            "# Анализ изображений",
            f"Всего изображений: {stats.total_images}",
            "",
            "## Основные характеристики",
            f"- Средний размер: {stats.basic_stats['avg_width']:.0f}x{stats.basic_stats['avg_height']:.0f}",
            f"- Минимальный размер: {stats.basic_stats['min_width']}x{stats.basic_stats['min_height']}",
            f"- Максимальный размер: {stats.basic_stats['max_width']}x{stats.basic_stats['max_height']}",
            "",
            "## Дубликаты",
            f"- Точные дубликаты: {len(stats.duplicates['exact'])} групп",
            f"- Приближенные дубликаты: {len(stats.duplicates['near'])} групп",
            "",
            "## Качество изображений",
            f"- Слишком темные: {len(stats.quality_issues['dark'])}",
            f"- Низкоконтрастные: {len(stats.quality_issues['low_contrast'])}",
            f"- Размытые: {len(stats.quality_issues['blurred'])}"
        ]

        with open(output_path, 'w', encoding='utf-8-sig') as f:
            f.write("\n".join(report))

    @staticmethod
    def generate_excel(stats: ImageStats, output_path: Path) -> None:
        """Генерация отчета в формате Excel"""
        # Основные метрики
        basic_df = pd.DataFrame([stats.basic_stats])
        
        # Качество
        quality_data = {
            'Тип проблемы': list(stats.quality_issues.keys()),
            'Количество': [len(v) for v in stats.quality_issues.values()]
        }
        quality_df = pd.DataFrame(quality_data)
        
        # Дубликаты
        duplicates_data = {
            'Тип дубликата': ['Точные', 'Приближенные'],
            'Количество групп': [
                len(stats.duplicates['exact']),
                len(stats.duplicates['near'])
            ]
        }
        duplicates_df = pd.DataFrame(duplicates_data)
        
        with pd.ExcelWriter(output_path) as writer:
            basic_df.to_excel(writer, sheet_name='Основные метрики', index=False)
            quality_df.to_excel(writer, sheet_name='Качество', index=False)
            duplicates_df.to_excel(writer, sheet_name='Дубликаты', index=False)