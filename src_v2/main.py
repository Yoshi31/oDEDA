import dataset
import analyzer
import stats
from pathlib import Path

def main():
    # Инициализация
    dataseta = dataset.ImageDataset()
    analyzera = analyzer.ImageAnalyzer(dataset)
    
    # Анализ
    quality_issues = analyzer.analyze_quality()
    statsa = stats.ImageStats(
        basic_stats=dataset.get_basic_stats(),
        quality_issues=quality_issues,
        duplicates=analyzer.find_duplicates()
    )
    
    stats.ReportGenerator.generate_markdown(statsa, Path("reports/report.md"))
    stats.ReportGenerator.generate_excel(statsa, Path("reports/stats.xlsx"))

if __name__ == "__main__":
    main()