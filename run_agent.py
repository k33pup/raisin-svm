import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import sklearn

from evaluate import check_predictions, dataset_labels
from tools import ask_model, read_file, read_pdf, run_python, write_file


RECIPE_PROMPT = """Read the supplied PDF pages as evidence. Extract only the selected paper experiment.
Return a JSON object with keys: rows (integer), features (integer), classes (list of names),
folds (integer), kernel (string), accuracy_percent (number, in percent), and sources.
sources is an object with keys dataset, features, folds, kernel, accuracy.
Each value is {"page": PDF page number, "quote": a short exact quote from that page}.
The dataset quote must contain the sample count, the features quote the feature count,
and the folds quote the fold count. For accuracy use a sentence from the results
that explicitly names SVM and its percentage. Use PDF page numbers shown in the input.
Return JSON only. Keep the paper's facts distinct from the task's implementation choices."""

CODE_PROMPT = """Write a complete runnable Python script for the supplied protocol. Return Python only.
numpy, pandas and scikit-learn are installed. Read raisin.csv in the current directory with pandas.
Class is the label. Its values Kecimen and Besni are strings. Keep the predicted strings, no int casting.
Use all data and the fixed protocol. Create a fresh StandardScaler + SVC Pipeline inside every fold.
Use SVC(kernel="poly", C=1.0, degree=3, gamma="scale", coef0=0.0). The kernel argument is mandatory.
Use StratifiedKFold with the extracted fold count, shuffle=True and random_state=42.
Use the held-out rows only for predict, never for fit. Print one FOLD line per fold with train/test sizes
and correct count. The input CSV has no row_id column: use the test indices returned by StratifiedKFold.
Collect (int(i), fold_number, str(label)) for i, label in zip(test_indices, predicted_labels).
Write pd.DataFrame(records, columns=["row_id", "fold", "prediction"]).to_csv("predictions.csv", index=False).
Include every held-out row exactly once; fold numbers start at 1.
Print one final line: print("METRIC", json.dumps({"accuracy": correct / total})).
Compute counts from the actual predictions. Use relative paths. No network or additional output files."""

SOURCE_FIELDS = ("dataset", "features", "folds", "kernel", "accuracy")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(text):
    return " ".join(text.split())


class Trace:
    def __init__(self, log, max_steps):
        self.log, self.max_steps = log, max_steps
        self.step = 0

    def event(self, **fields):
        print(json.dumps(fields, ensure_ascii=False), file=self.log, flush=True)

    def call(self, name, args, func, *params, summarize=lambda value: value):
        if self.step == self.max_steps:
            raise ValueError("Step limit reached")
        self.step += 1
        print(f"{self.step:02d} {name}", flush=True)
        try:
            result = func(*params)
        except Exception as error:
            self.event(step=self.step, tool=name, args=args, error=str(error))
            raise
        self.event(step=self.step, tool=name, args=args, result=summarize(result))
        return result


def validate_recipe(response, pages, trace):
    recipe = json.loads(response)
    if not isinstance(recipe, dict):
        raise ValueError("Recipe must be a JSON object")
    for name in ("rows", "features", "folds"):
        if type(recipe[name]) is not int or recipe[name] <= 0:
            raise ValueError(f"Invalid {name}")
    if not isinstance(recipe["classes"], list) or recipe["kernel"].casefold() not in ("poly", "polynomial"):
        raise ValueError("Expected the classes and the polynomial kernel of the selected experiment")

    corrections = []
    for name in SOURCE_FIELDS:
        source = recipe["sources"][name]
        index, quote = source["page"], compact(source["quote"])
        if type(index) is not int or not 1 <= index <= len(pages) or len(quote) < 8:
            raise ValueError(f"Invalid source for {name}")
        if quote not in compact(pages[index - 1]):
            matches = [i for i, text in enumerate(pages, 1) if quote in compact(text)]
            if len(matches) != 1:
                raise ValueError(f"Quote for {name} is not on page {index} or on one identifiable page: {quote}")
            source["page"] = matches[0]
            corrections.append({"field": name, "from_page": index, "to_page": matches[0]})

    for field, value in (("dataset", recipe["rows"]), ("features", recipe["features"]),
                         ("folds", recipe["folds"])):
        if not re.search(rf"\b{value}\b", recipe["sources"][field]["quote"]):
            raise ValueError(f"The {field} quote must contain the extracted count {value}")
    if "polynomial" not in recipe["sources"]["kernel"]["quote"].casefold():
        raise ValueError("Cite the paper's statement about the polynomial kernel")

    number = float(recipe["accuracy_percent"])
    quote = compact(recipe["sources"]["accuracy"]["quote"])
    if not 0 < number <= 100 or "SVM" not in quote or not re.search(rf"\b{re.escape(f'{number:g}')}\s*%", quote):
        raise ValueError("Cite a results sentence naming SVM and its actual accuracy in percent")
    recipe["accuracy_percent"] = number
    for correction in corrections:
        trace.event(event="citation_relocated", **correction)
    return recipe


def read_recipe(trace, ask, selector, pages, text):
    prompt = f"Selection and local protocol:\n{selector}\n\nSource pages:\n{text}"
    last_error = ""
    for _ in range(3):
        response = trace.call("llm_recipe", {"pages_in_prompt": text.count("=== PDF page ")},
                              ask, RECIPE_PROMPT, prompt, 1300)
        try:
            return validate_recipe(response, pages, trace)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            last_error = str(error)
            trace.event(event="recipe_rejected", error=last_error)
            prompt = (f"Selection and local protocol:\n{selector}\n\nSource pages:\n{text}\n\n"
                      f"Previous JSON:\n{response}\nRejected: {last_error}. Correct the evidence and return the full JSON.")
    raise ValueError(last_error)


def recipe_md(recipe, selector):
    pages = {field: source["page"] for field, source in recipe["sources"].items()}
    method_pages = ", ".join(map(str, sorted({pages["kernel"], pages["folds"]})))
    accuracy = f"{recipe['accuracy_percent']:g}".replace(".", ",")
    local = next(line for line in selector.splitlines() if line.startswith("Условия нашего запуска"))
    settings = [sentence for sentence in local.split(". ") if sentence.startswith(("Используем", "Для полиномиального"))]
    return "\n".join([
        "# Рецепт эксперимента", "",
        f"Из UCI беру таблицу на {recipe['rows']} объектов двух сортов ({', '.join(recipe['classes'])}). "
        f"Вход модели — {recipe['features']} признаков формы. В PDF данные описаны на стр. {pages['dataset']}, "
        f"признаки — на стр. {pages['features']}.", "",
        f"В статье используют SVM с полиномиальным ядром и {recipe['folds']} фолдов "
        f"(стр. {method_pages} PDF). На каждом шаге один фолд служит тестом, "
        "остальные идут в обучение. Accuracy — доля верных предсказаний.", "",
        f"У авторов {accuracy}% (таблица 6, стр. {pages['accuracy']} PDF). "
        "Точные цитаты, по которым агент собрал рецепт, есть в [логе](agent.log).", "",
        "## Настройки нашего прогона", "",
        "Точные разбиения и параметры ядра в статье не указаны. " + ". ".join(settings) + ".", "",
        "В каждом фолде обучаю `StandardScaler` и SVC только на тренировочных строках. "
        "Итог считаю по всем отложенным предсказаниям.", "",
    ])


def report_md(recipe, result, errors):
    value = recipe["accuracy_percent"]
    page = recipe["sources"]["accuracy"]["page"]
    lines = ["# SVM для двух сортов изюма", "", "## Воспроизведено", ""]
    if result:
        score = 100 * result["accuracy"]
        low, high = (100 * b for b in result["range"])
        lines += [
            "Обучил полиномиальный SVM на готовых признаках. Использовал 10 фолдов. "
            "Масштабирование обучал заново на тренировочных строках каждого фолда.", "",
            f"В статье {value:.2f}% (таблица 6, стр. {page} PDF). "
            f"В прогоне {score:.2f}%: {result['correct']} верных предсказаний из {result['total']}. "
            f"Допуск {low:.2f}–{high:.2f}%. "
            + ("Результат в диапазоне." if result["in_range"] else "Результат вне диапазона."), "",
            "[Предсказания](predictions.csv) · [Вывод Python](train.log) · [Рецепт](task.md)", "",
            "## Не воспроизведено", "",
            f"Расхождение со статьёй {score - value:+.2f} п.п. "
            f"Контрольный запуск с отдельной реализацией дал те же {result['total']} предсказаний. "
            "Точные фолды и настройки SVM авторы не опубликовали. "
            "В рецепте указаны наши фолды, параметры и масштабирование.", "",
        ]
    else:
        lines += [f"Рецепт и значение {value:.2f}% извлечены со стр. {page} PDF.", "",
                  "## Не воспроизведено", "", "Проверяемая accuracy не получена.", ""]
    if errors:
        lines += ["Ошибки запусков записаны в [agent.log](agent.log). "
                  f"Последняя ошибка — `{errors[-1].splitlines()[-1]}`.", ""]
    lines += ["## Не проверялось", "",
              "Съёмка изюма, выделение признаков из фотографий, другие модели и метрики статьи.", ""]
    return "\n".join(lines)


def reproduce(args, trace):
    task, out = args.task, args.output
    path = task / "task.md"
    selector = trace.call("read_file", {"path": str(path)}, read_file, path)
    pdf = task / "paper.pdf"
    pages = trace.call("read_pdf", {"path": str(pdf), "sha256": digest(pdf)}, read_pdf, pdf,
                       summarize=lambda pages: {"pages": len(pages), "characters": sum(map(len, pages))})
    blocks = [f"=== PDF page {i} ===\n{body}" for i, body in enumerate(pages, 1)]
    trace.call("write_file", {"path": str(out / "paper.txt")}, write_file,
               out / "paper.txt", "\n\n".join(blocks) + "\n")
    match = re.search(r"^Страницы PDF:\s*([\d, ]+)$", selector, re.M)
    positions = [int(n) for n in match[1].split(",")] if match else range(1, len(pages) + 1)
    if any(not 1 <= i <= len(pages) for i in positions):
        raise ValueError("Page selection falls outside the PDF")
    text = "\n\n".join(blocks[i - 1] for i in positions)

    def ask(system, prompt, tokens):
        return ask_model(args.base_url, args.model, system, prompt, tokens)

    recipe = read_recipe(trace, ask, selector, pages, text)
    trace.event(event="recipe_accepted", recipe=recipe)
    trace.call("write_file", {"path": str(out / "task.md")}, write_file,
               out / "task.md", recipe_md(recipe, selector))
    data_path = task / "data/raisin.csv"
    raw = read_file(data_path)
    header = next(csv.reader(raw.splitlines()))
    if len(dataset_labels(raw)) != recipe["rows"] or len(header) - 1 != recipe["features"]:
        raise ValueError("Data dimensions differ from extracted recipe")
    trace.event(event="dataset", path=str(data_path), sha256=digest(data_path), columns=header)
    expected_path = task / "expected.json"
    expected = json.loads(trace.call("read_file", {"path": str(expected_path)}, read_file, expected_path))
    low, high = expected["min"], expected["max"]
    if expected["metric"] != "accuracy" or not 0 <= low <= high <= 1:
        raise ValueError("Invalid expected accuracy range")
    protocol = f"{selector}\nExtracted fold count: {recipe['folds']}. Data columns: {header}."
    prompt = protocol
    script = out / "code/experiment.py"
    result = None
    errors = []
    (out / "train.log").touch()
    while trace.step + 6 <= trace.max_steps:
        source = trace.call("llm_code", {"attempt": len(errors) + 1}, ask, CODE_PROMPT, prompt, 1800)
        trace.call("write_file", {"path": str(script)}, write_file, script, source + "\n")
        run, predictions = trace.call(
            "run_python", {"path": str(script), "sha256": digest(script), "timeout_s": args.timeout},
            run_python, script, data_path, args.timeout,
            summarize=lambda value: {**value[0], "prediction_rows": max(0, value[1].count("\n") - 1)},
        )
        with (out / "train.log").open("a", encoding="utf-8") as train_log:
            train_log.write(f"Attempt {len(errors) + 1} | exit {run['exit_code']} | {run['elapsed_s']} s\n")
            train_log.write(run["stdout"])
            if run["stderr"]:
                train_log.write("\nstderr:\n" + run["stderr"])
            train_log.write("\n")
        try:
            if run["exit_code"]:
                raise ValueError("Timeout" if run["timed_out"] else
                                 (run["stderr"].strip() or f"Python exit {run['exit_code']}"))
            result = check_predictions(raw, predictions, run["stdout"], recipe, expected)
        except (ValueError, KeyError, TypeError) as error:
            errors.append(str(error))
            trace.event(event="experiment_rejected", error=str(error))
            prompt = (f"{protocol}\n\nPrevious script:\n{source}\n\n"
                      f"Run output:\n{run['stdout'][-3000:]}\nError:\n{str(error)[-3500:]}\n"
                      "Fix the failure. Preserve the protocol and return the full corrected script.")
            continue
        trace.event(event="checked", result=result)
        trace.call("write_file", {"path": str(out / "predictions.csv")}, write_file,
                   out / "predictions.csv", predictions)
        trace.call("write_file", {"path": str(out / "metrics.json")}, write_file,
                   out / "metrics.json", json.dumps(result, indent=2) + "\n")
        break
    trace.call("write_file", {"path": str(out / "REPORT.md")}, write_file,
               out / "REPORT.md", report_md(recipe, result, errors))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8094/v1"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "qwen3-4b"))
    parser.add_argument("--max-steps", type=int, default=21)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if args.max_steps < 12 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Use at least 12 steps and a positive timeout")
    if sys.platform != "darwin":
        parser.error("This runner uses macOS sandbox-exec")
    if args.output is None:
        args.output = Path("runs") / (args.task.name + datetime.now().strftime("-%Y%m%d-%H%M%S-%f"))
    args.output.mkdir(parents=True)
    with (args.output / "agent.log").open("w", encoding="utf-8") as log:
        trace = Trace(log, args.max_steps)
        trace.event(event="start", model=args.model, max_steps=args.max_steps, timeout_s=args.timeout,
                    python=sys.version.split()[0], sklearn=sklearn.__version__)
        try:
            result = reproduce(args, trace)
        except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
            trace.event(status="error", error=str(error))
            print(str(error), file=sys.stderr)
            return 1
        trace.event(status="complete" if result else "no validated result", steps=trace.step, result=result)
    return 0 if result else 1


if __name__ == "__main__":
    raise SystemExit(main())
