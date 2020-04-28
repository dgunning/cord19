from .cord19 import ResearchPapers
from .jsonpaper import get_json_paths
from pathlib import Path
import shutil


def export(papers: ResearchPapers, dest):
    dest_path = Path(dest)
    assert dest.exists(), f'Destination {dest} does not exist'
    paths = get_json_paths(papers.metadata, papers.data_path, first=False, tolist=True)
    print('Exporting', len(paths), 'json files to', dest_path)
    for src_file in paths:
        dest_file = dest_path / src_file
        dest_dir = dest_file.parent
        if not dest_dir.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_file, dest_file)