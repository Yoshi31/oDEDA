import os
import shutil
from tqdm import tqdm
from .stats import ImageStatsCollector

class DatasetOrganizer:
    def __init__(
        self,
        stats_collector: ImageStatsCollector,
        output_root: str,
        keep_originals: bool = True
    ):
        self.stats = stats_collector.compute_stats()
        self.output_root = output_root
        self.keep_originals = keep_originals
        self.stats_collector = stats_collector  # Сохраняем ссылку на коллектор
        
        # Создаем структуру папок
        self.folders = {
            'duplicates': os.path.join(output_root, 'duplicates'),
            'dark': os.path.join(output_root, 'dark'),
            'blurred': os.path.join(output_root, 'blurred'),
            'good': os.path.join(output_root, 'good')
        }
        
        for folder in self.folders.values():
            os.makedirs(folder, exist_ok=True)

    def organize(self):
        """Основной метод для сортировки изображений"""
        self._handle_duplicates()
        self._handle_low_quality()
        self._move_good_images()

    def _handle_duplicates(self):
        """Обработка дубликатов используя предварительно вычисленные данные"""
        # Используем сохраненные группы дубликатов из статистики
        for paths in tqdm(self.stats.duplicate_groups.values(), desc="Обработка дубликатов"):
            if len(paths) > 1:
                # Оставляем первый файл как оригинал
                original = paths[0]
                for duplicate in paths[1:]:
                    self._move_file(duplicate, self.folders['duplicates'])

    def _handle_low_quality(self):
        """Перемещает некачественные изображения используя готовые списки"""
        # Перемещаем темные изображения
        for path in tqdm(self.stats.dark_images, desc="Темные изображения"):
            self._move_file(path, self.folders['dark'])
        
        # Перемещаем размытые изображения
        for path in tqdm(self.stats.blurred_images, desc="Размытые изображения"):
            self._move_file(path, self.folders['blurred'])

    def _move_good_images(self):
        """Перемещает хорошие изображения в финальную папку"""
        all_images = set(self.stats_collector.dataset.image_paths)
        bad_images = set(
            self.stats.blurred_images +
            self.stats.dark_images +
            self.stats.low_contrast_images +
            [p for group in self.stats.duplicate_groups.values() for p in group[1:]]  # Все дубликаты кроме первого
        )
        good_images = all_images - bad_images
        
        for path in tqdm(good_images, desc="Качественные изображения"):
            self._move_file(path, self.folders['good'])

    def _move_file(self, src_path: str, dest_folder: str):
        """Универсальный метод для перемещения файлов"""
        if not os.path.exists(src_path):
            return

        filename = os.path.basename(src_path)
        dest_path = os.path.join(dest_folder, filename)
        
        # Обрабатываем коллизии имен
        counter = 1
        while os.path.exists(dest_path):
            name, ext = os.path.splitext(filename)
            dest_path = os.path.join(dest_folder, f"{name}_{counter}{ext}")
            counter += 1
        
        # Перемещаем файл
        shutil.move(src_path, dest_path)
        
        # Если нужно сохранить оригиналы - копируем обратно
        if self.keep_originals:
            shutil.copy(dest_path, src_path)
