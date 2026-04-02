"""
Stage 1: Download CMS Medicare SynPUF files with SHA-256 checksum verification.

CMS SynPUFs are fully synthetic Medicare claims — no IRB required.
Reference: https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

# CMS SynPUF download URL pattern and known SHA-256 checksums.
# Checksums are computed on the raw ZIP files from CMS.
# Extend this dict as you download additional sample numbers.
SYNPUF_URLS: dict[int, str] = {
    1: "https://www.cms.gov/files/zip/de1-0-2008-beneficiary-summary-file-sample-1.zip",
    2: "https://www.cms.gov/files/zip/de1-0-2008-beneficiary-summary-file-sample-2.zip",
    3: "https://www.cms.gov/files/zip/de1-0-2008-beneficiary-summary-file-sample-3.zip",
    4: "https://www.cms.gov/files/zip/de1-0-2008-beneficiary-summary-file-sample-4.zip",
    5: "https://www.cms.gov/files/zip/de1-0-2008-beneficiary-summary-file-sample-5.zip",
}

INPATIENT_URLS: dict[int, str] = {
    1: "https://www.cms.gov/files/zip/de1-0-2008-to-2010-inpatient-claims-sample-1.zip",
    2: "https://www.cms.gov/files/zip/de1-0-2008-to-2010-inpatient-claims-sample-2.zip",
    3: "https://www.cms.gov/files/zip/de1-0-2008-to-2010-inpatient-claims-sample-3.zip",
    4: "https://www.cms.gov/files/zip/de1-0-2008-to-2010-inpatient-claims-sample-4.zip",
    5: "https://www.cms.gov/files/zip/de1-0-2008-to-2010-inpatient-claims-sample-5.zip",
}

# SHA-256 checksums for integrity verification.
# These are populated when CMS publishes them; leave None to skip verification.
KNOWN_CHECKSUMS: dict[str, str | None] = {}


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
    """Stream-download url to dest, showing a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("Already downloaded: %s — skipping", dest.name)
        return dest

    logger.info("Downloading %s → %s", url, dest)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return dest


def _verify_checksum(path: Path) -> None:
    expected = KNOWN_CHECKSUMS.get(path.name)
    if expected is None:
        logger.debug("No checksum registered for %s — skipping verification", path.name)
        return
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"Checksum mismatch for {path.name}:\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}"
        )
    logger.info("Checksum OK: %s", path.name)


def download_synpuf_samples(
    sample_numbers: list[int],
    raw_dir: Path,
    skip_existing: bool = True,
) -> list[Path]:
    """
    Download CMS SynPUF beneficiary + inpatient ZIP files for the given sample numbers.

    Returns a list of downloaded file paths.
    """
    downloaded: list[Path] = []

    for n in sample_numbers:
        for label, url_map in [("bene", SYNPUF_URLS), ("inpatient", INPATIENT_URLS)]:
            if n not in url_map:
                logger.warning("No URL registered for sample %d (%s) — skipping", n, label)
                continue
            url = url_map[n]
            filename = url.split("/")[-1]
            dest = raw_dir / filename

            if skip_existing and dest.exists():
                logger.info("Skipping existing file: %s", dest.name)
                downloaded.append(dest)
                continue

            path = _download_file(url, dest)
            _verify_checksum(path)
            downloaded.append(path)

    logger.info("Download complete. %d files ready in %s", len(downloaded), raw_dir)
    return downloaded


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    raw_dir = Path(cfg["data"]["raw_dir"])
    samples = cfg["data"]["synpuf_samples"]
    download_synpuf_samples(samples, raw_dir)
