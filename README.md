# Итоговая работа

Доработал агента из ДЗ-3. Он читает статью, собирает рецепт одного эксперимента и запускает написанный моделью код. В примере повторил SVM для двух сортов изюма.

## Запуск

Нужны macOS, Python 3.13 и OpenAI-совместимый сервер. PDF и данные уже лежат в `text_tasks/raisin-svm/`.

Из корня репозитория установите зависимости и запустите llama.cpp.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
bash serve_local.sh
```

Во втором терминале из той же папки запустите агента.

```bash
.venv/bin/python run_agent.py --task text_tasks/raisin-svm
```

`serve_local.sh` ищет сервер и веса Qwen3-4B-Instruct-2507 Q4_K_M в `../.runtime` относительно корня репозитория. Другие пути можно задать через `LLAMA_SERVER` и `LLAMA_MODEL_PATH`. Для уже запущенного API укажите `--base-url` (адрес с `/v1`) и `--model`.

Лимит — 21 шаг (`--max-steps`), таймаут Python — 30 секунд (`--timeout`). Новые прогоны сохраняются в `runs/`.

Сохранённый прогон в `demo/`. [Отчёт](demo/REPORT.md) · [Рецепт](demo/task.md) · [Лог агента](demo/agent.log)
