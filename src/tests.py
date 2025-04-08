import os
import cv2
import numpy as np
from PIL import Image
import imagehash
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import pandas as pd
import hashlib

# Установка бэкенда Matplotlib
import matplotlib
matplotlib.use('Agg')

class ImageDataset:
    def __init__(self, image_dir: str, img_extensions: List[str] = None, num_workers: int = 4):
        self.image_dir = image_dir
        self.img_extensions = img_extensions or ['.jpg', '.jpeg', '.png', '.bmp']
        self.num_workers = num_workers
        self.image_paths = self._get_image_paths()
        self.metadata = self._load_metadata()

    def _get_image_paths(self) -> List[str]:
        """Рекурсивно собирает пути к изображениям с использованием многопоточной обработки."""
        paths = []
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.img_extensions):
                    paths.append(os.path.join(root, file))
        return paths

    def _load_metadata(self) -> List[Dict]:
        """Параллельная загрузка метаданных с прогресс-баром."""
        def process_image(path):
            try:
                with Image.open(path) as img:
                    width, height = img.size
                    hashes = {
                        'phash': str(imagehash.phash(img)),
                        'whash': str(imagehash.whash(img)),
                        'md5': hashlib.md5(np.array(img).tobytes()).hexdigest()
                    }
                    return {
                        "path": path,
                        "width": width,
                        "height": height,
                        "hashes": hashes
                    }
            except Exception:
                return None

        with ThreadPoolExecutor(self.num_workers) as executor:
            results = list(tqdm(
                executor.map(process_image, self.image_paths),
                total=len(self.image_paths),
                desc="Загрузка метаданных"
            ))
        
        return [r for r in results if r is not None]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> np.ndarray:
        return cv2.imread(self.image_paths[idx])

@dataclass
class ImageStats:
    total_images: int
    avg_width: float
    avg_height: float
    min_width: int
    max_width: int
    min_height: int
    max_height: int
    exact_duplicates: Dict[str, List[str]] = field(default_factory=dict)
    near_duplicates: Dict[str, List[str]] = field(default_factory=dict)
    brightness_stats: Dict[str, float] = field(default_factory=dict)
    contrast_stats: Dict[str, float] = field(default_factory=dict)
    sharpness_stats: Dict[str, float] = field(default_factory=dict)
    quality_issues: Dict[str, List[str]] = field(default_factory=dict)

class ImageAnalyzer:
    def __init__(self, dataset: ImageDataset, num_workers: int = 4):
        self.dataset = dataset
        self.num_workers = num_workers
        self._quality_metrics = {}
        self._similarity_cache = {}

    def compute_stats(self) -> ImageStats:
        """Основной метод для вычисления всей статистики."""
        # Базовые метрики
        basic_stats = self._compute_basic_stats()
        
        # Поиск дубликатов
        exact_dups, near_dups = self._find_all_duplicates()
        
        # Анализ качества
        quality_metrics = self._analyze_quality()
        
        return ImageStats(
            total_images=len(self.dataset),
            avg_width=basic_stats['avg_width'],
            avg_height=basic_stats['avg_height'],
            min_width=basic_stats['min_width'],
            max_width=basic_stats['max_width'],
            min_height=basic_stats['min_height'],
            max_height=basic_stats['max_height'],
            exact_duplicates=exact_dups,
            near_duplicates=near_dups,
            brightness_stats={
                'mean': np.mean([m['brightness'] for m in quality_metrics.values()]),
                'std': np.std([m['brightness'] for m in quality_metrics.values()])
            },
            contrast_stats={
                'mean': np.mean([m['contrast'] for m in quality_metrics.values()]),
                'std': np.std([m['contrast'] for m in quality_metrics.values()])
            },
            sharpness_stats={
                'mean': np.mean([m['sharpness'] for m in quality_metrics.values()]),
                'std': np.std([m['sharpness'] for m in quality_metrics.values()])
            },
            quality_issues=self._detect_quality_issues(quality_metrics)
        )

    def _compute_basic_stats(self) -> Dict[str, float]:
        """Вычисление базовой статистики по размерам."""
        widths = [meta["width"] for meta in self.dataset.metadata]
        heights = [meta["height"] for meta in self.dataset.metadata]
        
        return {
            'avg_width': np.mean(widths),
            'avg_height': np.mean(heights),
            'min_width': min(widths),
            'max_width': max(widths),
            'min_height': min(heights),
            'max_height': max(heights)
        }

    def _find_all_duplicates(self) -> Tuple[Dict, Dict]:
        """Поиск точных и приближенных дубликатов."""
        # Группировка по хешам
        hash_groups = {'phash': {}, 'whash': {}, 'md5': {}}
        
        for meta in self.dataset.metadata:
            for hash_type in hash_groups:
                h = meta['hashes'][hash_type]
                hash_groups[hash_type].setdefault(h, []).append(meta['path'])
        
        # Точные дубликаты (совпадение по всем хешам)
        exact_duplicates = {}
        for h, paths in hash_groups['md5'].items():
            if len(paths) > 1:
                exact_duplicates[h] = paths
        
        # Приближенные дубликаты (по perceptual hashes)
        near_duplicates = self._find_near_duplicates(hash_groups['phash'])
        
        return exact_duplicates, near_duplicates

    def _find_near_duplicates(self, phash_groups: Dict) -> Dict:
        """Поиск приближенных дубликатов с учетом порога схожести."""
        near_duplicates = {}
        hashes = list(phash_groups.keys())
        
        # Оптимизация: сравниваем только потенциально похожие хеши
        for i in tqdm(range(len(hashes)), desc="Поиск near-duplicates"):
            for j in range(i+1, len(hashes)):
                h1, h2 = hashes[i], hashes[j]
                distance = self._hamming_distance(h1, h2)
                
                if distance <= 5:  # Порог можно настраивать
                    key = f"{h1}_{h2}"
                    near_duplicates[key] = phash_groups[h1] + phash_groups[h2]
        
        return near_duplicates

    def _hamming_distance(self, h1: str, h2: str) -> int:
        """Вычисление расстояния Хэмминга между хешами."""
        return sum(c1 != c2 for c1, c2 in zip(h1, h2))

    def _analyze_quality(self) -> Dict[str, Dict[str, float]]:
        """Многопоточный анализ качества изображений."""
        def process_image(idx):
            img = self.dataset[idx]
            if img is None:
                return None
                
            # Яркость
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            brightness = np.mean(hsv[:, :, 2])
            
            # Контраст
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            contrast = np.std(gray)
            
            # Резкость
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = np.var(laplacian)
            
            return {
                'brightness': brightness,
                'contrast': contrast,
                'sharpness': sharpness
            }

        with ThreadPoolExecutor(self.num_workers) as executor:
            results = list(tqdm(
                executor.map(process_image, range(len(self.dataset))),
                total=len(self.dataset),
                desc="Анализ качества"
            ))
        
        return {
            self.dataset.image_paths[i]: r 
            for i, r in enumerate(results) 
            if r is not None
        }

    def _detect_quality_issues(self, metrics: Dict) -> Dict[str, List[str]]:
        """Выявление проблемных изображений."""
        issues = {
            'dark': [],
            'low_contrast': [],
            'blurred': []
        }
        
        brightness = [m['brightness'] for m in metrics.values()]
        contrast = [m['contrast'] for m in metrics.values()]
        sharpness = [m['sharpness'] for m in metrics.values()]
        
        b_mean, b_std = np.mean(brightness), np.std(brightness)
        c_mean, c_std = np.mean(contrast), np.std(contrast)
        s_mean, s_std = np.mean(sharpness), np.std(sharpness)
        
        for path, m in metrics.items():
            if m['brightness'] < b_mean - 2*b_std:
                issues['dark'].append(path)
            if m['contrast'] < c_mean - 2*c_std:
                issues['low_contrast'].append(path)
            if m['sharpness'] < s_mean - 2*s_std:
                issues['blurred'].append(path)
                
        return issues

    def generate_report(self, stats: ImageStats, report_path: str = None) -> str:
        """Генерация отчета в формате Markdown."""
        report = [
            "# Analys images",
            f"Total images: {stats.total_images}",
            "",
            "## Main",
            f"- Average size: {stats.avg_width:.0f}x{stats.avg_height:.0f}",
            f"- Min size: {stats.min_width}x{stats.min_height}",
            f"- Max size: {stats.max_width}x{stats.max_height}",
            "",
            "## Dublicates",
            f"- dublicates: {len(stats.exact_duplicates)} groups",
            f"- near Dublicates: {len(stats.near_duplicates)} groups",
            "",
            "## quality images",
            f"- dark: {len(stats.quality_issues['dark'])}",
            f"- low contrast: {len(stats.quality_issues['low_contrast'])}",
            f"- blured: {len(stats.quality_issues['blurred'])}"
        ]
        
        report_text = "\n".join(report)
        
        if report_path:
            with open(report_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    

    # Инициализация
dataset = ImageDataset("C:/Users/zayka/ML_Project/oDEDA/dataset_example/train/images", num_workers=16)
analyzer = ImageAnalyzer(dataset)

# Анализ
stats = analyzer.compute_stats()

# Генерация отчета
report = analyzer.generate_report(stats, "report.md")

# Просмотр результатов
print(f"Найдено {len(stats.exact_duplicates)} групп точных дубликатов")
print(f"Найдено {len(stats.near_duplicates)} групп near-duplicates")
# print(f"Проблемные изображения: {stats.quality_issues}")