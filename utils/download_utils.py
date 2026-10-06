import sys
from pathlib import Path
from urllib.parse import urljoin

from torch.hub import download_url_to_file


def load_file_from_url(
    url: str,
    model_dir: Path | None = None,
    progress: bool = True,
    file_name: str | None = None,
) -> str:
    """Load file form http url, will download models if necessary."""
    file_name = Path(file_name)
    print(f"[ProPainter] Checking/Creating model directory: {model_dir}", file=sys.stderr, flush=True)
    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)
    cached_file = model_dir / file_name
    print(f"[ProPainter] Checking if model exists: {cached_file}", file=sys.stderr, flush=True)
    if not cached_file.exists():
        print(f'[ProPainter] Downloading: "{url}" to {cached_file}', file=sys.stderr, flush=True)
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return str(cached_file)


def download_model(model_url: str, model_name: str) -> str:
    """Downloads a model from a URL and returns the local path to the downloaded model."""
    base_dir = Path(__file__).parents[1].absolute()
    target_dir = base_dir / "weights"
    return load_file_from_url(
        url=urljoin(model_url, model_name),
        model_dir=target_dir,
        progress=True,
        file_name=model_name,
    )
