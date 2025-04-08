from data import images
from data import stats
from data import organizer

# Инициализация датасета и сбор статистики
dataset = images.ImageDataset("C:/Users/zayka/ML_Project/oDEDA/dataset_example/train/images")
collector = stats.ImageStatsCollector(dataset)
    
# Получение статистики
stats = collector.compute_stats()
# print(stats.duplicate_count)
# # Выводим список дубликатов
# print("\n=== ГРУППЫ ДУБЛИКАТОВ ===")
# for hash_value, paths in stats.duplicate_groups.items():
#     print(f"\nХэш: {hash_value}")
#     print(f"Количество дубликатов: {len(paths)-1}")
#     print("Файлы:")
#     for i, path in enumerate(paths, 1):
#         print(f"{i}. {path}")
# #print(stats)
# print(f"Средняя яркость: {stats.brightness_mean:.1f}")
# print(f"Размытых изображений: {len(stats.blurred_images)}")

# # Визуализация
# collector.plot_quality_distribution("quality_plot.png")

# # Фильтрация некачественных
# low_quality = collector.get_low_quality_images()
# print("/nПримеры темных изображений:", low_quality["dark"][:3])
# print("/nПримеры темных изображений:", low_quality["low_contrast"][:3])
# print("/nПримеры темных изображений:", low_quality["blurred"][:3])


    
# # Сортировка
organizer = organizer.DatasetOrganizer(
    stats_collector=collector,
    output_root="C:/Users/zayka/ML_Project/oDEDA/sorted",
    keep_originals=False
)
organizer.organize()