# Данные

Raisin Dataset, Çınar, Köklü и Taşdemir. [UCI Machine Learning Repository](https://doi.org/10.24432/C5660T), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

`raisin.csv` получен из официального ARFF. Значения, порядок строк и метки взяты из исходного файла. Числовой индекс строки отсчитывается с нуля.

Повторить экспорт из корня репозитория.

```bash
curl -fL https://archive.ics.uci.edu/static/public/850/raisin.zip -o raisin.zip
python prepare_data.py raisin.zip text_tasks/raisin-svm/data/raisin.csv
```
