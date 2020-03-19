import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Collection, Any
from tqdm import tqdm_notebook as tqdm
import time
import multiprocessing
from jinja2 import Template
from functools import lru_cache


def num_cpus() -> int:
    "Get number of cpus"
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def ifnone(a: Any, b: Any) -> Any:
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a


@lru_cache(maxsize=16)
def load_template(template):
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    template_file = os.path.join(template_dir, f'{template}.template')
    with open(template_file, 'r') as f:
        return Template(f.read())


def render_html(template_name, **kwargs):
    template = load_template(template_name)
    return template.render(kwargs)


def parallel(func, arr: Collection, max_workers: int = None, leave=False):
    "Call `func` on every element of `arr` in parallel using `max_workers`."
    max_workers = ifnone(max_workers, multiprocessing.cpu_count())
    progress_bar = tqdm(arr)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures_to_index = {ex.submit(func, o):i for i, o in enumerate(arr)}
        results = []
        for f in as_completed(futures_to_index):
            results.append((futures_to_index[f], f.result()))
            progress_bar.update()
        for n in range(progress_bar.n, progress_bar.total):
            time.sleep(0.1)
            progress_bar.update()
        results.sort(key=lambda x: x[0])
    return [result for i, result in results]


def add(cat1, cat2):
    return cat1 + cat2