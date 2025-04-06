import os
import cv2
import numpy as np
from PIL import Image
import imagehash
from typing import List, Dict, Optional, Union
from tqdm import tqdm

class ImageDataset:
    def __init__(self, image_dir: str, img_extensions: List[str] = None):
        """
        Инициализация датасета.
        :param image_dir: Путь к папке с изображениями.
        :param img_extensions: Список расширений (например, ['.jpg', '.png']).
        """
        self.image_dir = image_dir
        self.img_extensions = img_extensions or ['.jpg', '.jpeg', '.png', '.bmp']
        self.image_paths = self._get_image_paths()
        self.metadata = self._load_metadata()

    def _get_image_paths(self) -> List[str]:
        """Возвращает список путей к изображениям в указанной папке."""
        paths = []
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.img_extensions):
                    paths.append(os.path.join(root, file))
        return paths

    def _load_metadata(self) -> List[Dict]:
        """Загружает метаданные для всех изображений (размеры, хеши)."""
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
        """Количество изображений в датасете."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> np.ndarray:
        """Возвращает изображение по индексу (как numpy array)."""
        return cv2.imread(self.image_paths[idx])

    def get_image(self, idx: int, as_pil: bool = False) -> Union[np.ndarray, Image.Image]:
        """Возвращает изображение в формате numpy (OpenCV) или PIL."""
        if as_pil:
            return Image.open(self.image_paths[idx])
        return self[idx]

    def get_metadata(self, idx: int) -> Dict:
        """Возвращает метаданные изображения по индексу."""
        return self.metadata[idx]


    def stats(self) -> Dict:
        """Возвращает статистику по датасету (размеры, уникальные хеши)."""
        widths = [meta["width"] for meta in self.metadata]
        heights = [meta["height"] for meta in self.metadata]
        unique_hashes = len(set(meta["hash"] for meta in self.metadata))
        return {
            "total_images": len(self),
            "unique_hashes": unique_hashes,
            "avg_width": np.mean(widths),
            "avg_height": np.mean(heights),
            "min_width": min(widths),
            "max_width": max(widths),
        }

    def __repr__(self) -> str:
        return f"ImageDataset(images={len(self)}, path='{self.image_dir}')"