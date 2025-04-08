import os
import cv2
import numpy as np
from PIL import Image
import imagehash
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
from tqdm import tqdm

# Установка бэкенда Matplotlib для избежания конфликтов с Qt
import matplotlib
matplotlib.use('Agg')

@dataclass
class ImageStats:
    total_images: int
    avg_width: float
    avg_height: float
    min_width: int
    max_width: int
    min_height: int
    max_height: int
    unique_hashes: int
    duplicate_count: int = 0
    duplicate_groups: dict = field(default_factory=dict)
    brightness_mean: float = 0.0
    brightness_std: float = 0.0
    contrast_mean: float = 0.0
    contrast_std: float = 0.0
    sharpness_mean: float = 0.0
    sharpness_std: float = 0.0
    quality_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    blurred_images: List[str] = field(default_factory=list)
    dark_images: List[str] = field(default_factory=list)
    low_contrast_images: List[str] = field(default_factory=list)

class ImageDataset:
    def __init__(self, image_dir: str, img_extensions: List[str] = None):
        self.image_dir = image_dir
        self.img_extensions = img_extensions or ['.jpg', '.jpeg', '.png', '.bmp']
        self.image_paths = self._get_image_paths()
        self.metadata = self._load_metadata()

    def _get_image_paths(self) -> List[str]:
        paths = []
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.img_extensions):
                    paths.append(os.path.join(root, file))
        return paths

    def _load_metadata(self) -> List[Dict]:
        metadata = []
        for path in tqdm(self.image_paths, desc="Загрузка метаданных"):
            with Image.open(path) as img:
                width, height = img.size
                img_hash = str(imagehash.phash(img))
            metadata.append({
                "path": path,
                "width": width,
                "height": height,
                "hash": img_hash,
            })
        return metadata

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> np.ndarray:
        return cv2.imread(self.image_paths[idx])

    def get_image(self, idx: int, as_pil: bool = False) -> Union[np.ndarray, Image.Image]:
        if as_pil:
            return Image.open(self.image_paths[idx])
        return self[idx]

class ImageStatsCollector:
    def __init__(self, dataset: ImageDataset):
        self.dataset = dataset
        self._stats: Optional[ImageStats] = None
        self._quality_metrics: Dict[str, Dict[str, float]] = {}
        self._compute_all_metrics()

    def _compute_all_metrics(self):
        """Вычисляет все метрики качества один раз при инициализации."""
        for idx in tqdm(range(len(self.dataset)), desc="Анализ качества"):
            img = self.dataset[idx]
            path = self.dataset.image_paths[idx]
            self._quality_metrics[path] = self._evaluate_image_quality(img)

    def _evaluate_image_quality(self, img: np.ndarray) -> Dict[str, float]:
        """Вычисляет метрики качества для одного изображения."""
        # Яркость (среднее значение V-канала в HSV)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2])
        
        # Контраст (стандартное отклонение в градациях серого)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        
        # Резкость (дисперсия оператора Лапласа)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        # Размытость (порог эмпирически подобран)
        is_blurred = sharpness < 100.0
        
        return {
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": sharpness,
            "is_blurred": is_blurred,
        }

    def compute_stats(self) -> ImageStats:
        """Собирает полную статистику о датасете."""
        # Основные метаданные
        widths = [meta["width"] for meta in self.dataset.metadata]
        heights = [meta["height"] for meta in self.dataset.metadata]
        hashes = [meta["hash"] for meta in self.dataset.metadata]
        
        # Поиск дубликатов
        hash_groups = {}
        for meta in self.dataset.metadata:
            hash_groups.setdefault(meta["hash"], []).append(meta["path"])
        duplicates = {h: paths for h, paths in hash_groups.items() if len(paths) > 1}
        
        # Качество изображений
        brightness = [m["brightness"] for m in self._quality_metrics.values()]
        contrast = [m["contrast"] for m in self._quality_metrics.values()]
        sharpness = [m["sharpness"] for m in self._quality_metrics.values()]
        
        # Фильтрация некачественных
        low_quality = self.get_low_quality_images()
        
        return ImageStats(
            total_images=len(self.dataset),
            avg_width=np.mean(widths),
            avg_height=np.mean(heights),
            min_width=min(widths),
            max_width=max(widths),
            min_height=min(heights),
            max_height=max(heights),
            unique_hashes=len(set(hashes)),
            duplicate_groups=duplicates,
            duplicate_count=sum(len(paths) - 1 for paths in duplicates.values()),
            brightness_mean=np.mean(brightness),
            brightness_std=np.std(brightness),
            contrast_mean=np.mean(contrast),
            contrast_std=np.std(contrast),
            sharpness_mean=np.mean(sharpness),
            sharpness_std=np.std(sharpness),
            quality_metrics=self._quality_metrics,
            blurred_images=low_quality["blurred"],
            dark_images=low_quality["dark"],
            low_contrast_images=low_quality["low_contrast"],
        )

    def get_low_quality_images(
        self,
        brightness_threshold: float = 30.0,
        contrast_threshold: float = 20.0,
    ) -> Dict[str, List[str]]:
        """Возвращает пути некачественных изображений по порогам."""
        dark = []
        low_contrast = []
        blurred = []
        
        for path, metrics in self._quality_metrics.items():
            if metrics["brightness"] < brightness_threshold:
                dark.append(path)
            if metrics["contrast"] < contrast_threshold:
                low_contrast.append(path)
            if metrics["is_blurred"]:
                blurred.append(path)
                
        return {
            "dark": dark,
            "low_contrast": low_contrast,
            "blurred": blurred,
        }

    def plot_quality_distribution(self, save_path: Optional[str] = None) -> plt.Figure:
        """Визуализирует распределение метрик качества."""
        brightness = [m["brightness"] for m in self._quality_metrics.values()]
        contrast = [m["contrast"] for m in self._quality_metrics.values()]
        sharpness = [m["sharpness"] for m in self._quality_metrics.values()]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].hist(brightness, bins=20, color='blue', alpha=0.7)
        axes[0].set_title("Яркость")
        axes[0].axvline(50, color='red', linestyle='--', label="Идеал (50)")
        
        axes[1].hist(contrast, bins=20, color='green', alpha=0.7)
        axes[1].set_title("Контраст")
        axes[1].axvline(30, color='red', linestyle='--', label="Идеал (30)")
        
        axes[2].hist(sharpness, bins=20, color='purple', alpha=0.7)
        axes[2].set_title("Резкость")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        return fig
