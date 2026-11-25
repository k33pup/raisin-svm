import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import urllib.request

from pypdf import PdfReader


def read_file(path):
    return path.read_text(encoding="utf-8")


def write_file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_pdf(path):
    pages = [page.extract_text().strip() for page in PdfReader(path).pages]
    if not pages or not any(pages):
        raise ValueError("The PDF has no text layer")
    return pages


def ask_model(base_url, model, system, prompt, max_tokens):
    body = {
        "model": model, "temperature": 0, "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    }
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        json.dumps(body).encode(), headers=headers,
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        choice = json.load(response)["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ValueError("The model response was truncated")
    answer = choice["message"]["content"].strip()
    if answer.startswith("```") and answer.endswith("```"):
        answer = answer.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    if not answer:
        raise ValueError("Empty model response")
    return answer


def limit_output():
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))


def run_python(script, data, timeout):
    start = time.monotonic()
    with TemporaryDirectory(prefix="raisin-run-") as folder:
        work = Path(folder).resolve()
        shutil.copyfile(script, work / "experiment.py")
        shutil.copyfile(data, work / "raisin.csv")
        prediction_file = work / "predictions.csv"

        readable = {work, Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
        paths = " ".join(f"(subpath {json.dumps(str(path))})" for path in sorted(readable))
        profile = (
            "(version 1)(allow default)(deny network*)(deny process-fork)(deny file-write*)"
            '(deny file-read-data (subpath "/Users") (subpath "/Volumes")'
            ' (subpath "/private/tmp") (subpath "/private/var/folders"))'
            f"(allow file-read-data {paths} (literal {json.dumps(str(Path(sys.executable).resolve()))}))"
            f"(allow file-write* (literal {json.dumps(str(prediction_file))}))"
        )
        with (work / "stdout").open("w+b") as stdout, (work / "stderr").open("w+b") as stderr:
            child = subprocess.Popen(
                ["/usr/bin/sandbox-exec", "-p", profile, sys.executable, "-I", "-B", "experiment.py"],
                cwd=work, env={"PATH": "/usr/bin:/bin", "LC_ALL": "C",
                               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                               "VECLIB_MAXIMUM_THREADS": "1"},
                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                start_new_session=True, preexec_fn=limit_output,
            )
            timed_out = False
            try:
                code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                timed_out, code = True, 124
            stdout.seek(0)
            stderr.seek(0)
            return {
                "exit_code": code, "timed_out": timed_out,
                "stdout": stdout.read().decode("utf-8", errors="replace"),
                "stderr": stderr.read().decode("utf-8", errors="replace"),
                "elapsed_s": round(time.monotonic() - start, 3),
            }, read_file(prediction_file) if prediction_file.is_file() else ""
