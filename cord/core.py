import concurrent
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace
from typing import Collection, Any
from tqdm import tqdm_notebook as tqdm
import time

def num_cpus() -> int:
    "Get number of cpus"
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def ifnone(a: Any, b: Any) -> Any:
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a

_default_cpus = min(16, num_cpus())
defaults = SimpleNamespace(cpus=_default_cpus, cmap='viridis', return_fig=False, silent=False)


def parallel(func, arr: Collection, max_workers: int = None, leave=False):
    "Call `func` on every element of `arr` in parallel using `max_workers`."
    max_workers = ifnone(max_workers, defaults.cpus)
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