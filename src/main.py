from data import images
from data import stats

# Инициализация датасета и сбор статистики
dataset = images.ImageDataset("/home/lcv/Work/odEDA/dataset_example/train/images")
collector = stats.ImageStatsCollector(dataset)
    
# Получение статистики
stats = collector.compute_stats()
print(f"Средняя яркость: {stats.brightness_mean:.1f}")
print(f"Размытых изображений: {len(stats.blurred_images)}")

# Визуализация
collector.plot_quality_distribution("quality_plot.png")

# Фильтрация некачественных
low_quality = collector.get_low_quality_images()
print("\nПримеры темных изображений:", low_quality["dark"][:3])