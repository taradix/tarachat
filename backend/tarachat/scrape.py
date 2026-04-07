"""Scrape the web for documents."""

import asyncio
import json
import logging
from pathlib import Path

import aiofiles
import aiohttp
from bs4 import BeautifulSoup
from yarl import URL

logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT = 60
DEFAULT_CHUNK_SIZE = 1024 * 1024


def meta_path_for(file_path: Path) -> Path:
    return file_path.with_suffix(file_path.suffix + ".meta.json")


def load_metadata(file_path: Path) -> dict:
    meta_file = meta_path_for(file_path)
    if meta_file.exists():
        try:
            with meta_file.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_metadata(file_path: Path, metadata: dict) -> None:
    meta_file = meta_path_for(file_path)
    with meta_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def has_changed(local_meta: dict, remote_meta: dict) -> bool:
    """Return True if we should re-download the file."""
    return any([
        not remote_meta,
        remote_meta["etag"] != local_meta.get("etag"),
        remote_meta["last_modified"] != local_meta["last_modified"],
        remote_meta["content_length"] != local_meta["content_length"],
    ])


async def fetch_remote_metadata(session: aiohttp.ClientSession, url: URL) -> dict:
    """Get metadata via HEAD. If it fails, return {}."""
    try:
        async with session.head(url, allow_redirects=True, timeout=20) as resp:
            # Some servers may not like HEAD; treat non-2xx as no metadata
            if resp.status >= 400:
                return {}
            headers = resp.headers
            return {
                "etag": headers.get("ETag"),
                "last_modified": headers.get("Last-Modified"),
                "content_length": headers.get("Content-Length"),
                "url": str(url),
            }
    except Exception:
        return {}


async def download_one(
    session: aiohttp.ClientSession,
    url: URL,
    target_dir: Path,
    timeout: int = DEFAULT_TIMEOUT,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[str, str | None, str]:
    """Download a single URL if it has changed."""
    filename = url.name
    file_path = target_dir / filename

    logger.info(f"[{url}] Checking remote metadata...")
    remote_meta = await fetch_remote_metadata(session, url)
    local_meta = load_metadata(file_path)

    if file_path.exists() and not has_changed(local_meta, remote_meta):
        logger.info(f"[{url}] Unchanged, skipping.")
        return (url, str(file_path), "skipped")

    logger.info(f"[{url}] Downloading to {file_path}...")
    try:
        async with session.get(url, timeout=timeout) as resp:
            resp.raise_for_status()

            async with aiofiles.open(file_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(chunk_size):
                    if chunk:
                        await f.write(chunk)

        if remote_meta:
            save_metadata(file_path, remote_meta)

        return (url, str(file_path), "downloaded")

    except Exception:
        logger.exception(f"[{url}] Download failed")
        return (url, None, "error")


async def download_many(
    urls: list[URL],
    target_dir: Path,
    max_concurrency: int = 5,
    timeout: int = DEFAULT_TIMEOUT,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[tuple[str, str | None, str]]:
    """Download multiple URLs concurrently if the remote file changed."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(max_concurrency)

    async with aiohttp.ClientSession() as session:

        async def worker(u: URL):
            async with semaphore:
                return await download_one(session, u, target_dir, timeout, chunk_size)

        tasks = [asyncio.create_task(worker(u)) for u in urls]
        return await asyncio.gather(*tasks)


async def get_urls(url: URL, timeout: int = DEFAULT_TIMEOUT) -> list[URL]:
    async with aiohttp.ClientSession() as session, session.get(url, timeout=timeout) as resp:
        resp.raise_for_status()
        data = await resp.json()
        html_content = data['contenu']
        soup = BeautifulSoup(html_content, "html.parser")
        return [URL(a.get("href")) for a in soup.find_all("a")]


async def _async_main():
    url = URL("https://vplus.modellium.com/api/www.notre-dame-du-laus.ca/structure/detail/reglements?localisation=fr")
    urls = await get_urls(url)
    target_dir = Path("data/documents")
    await download_many(urls, target_dir)


def main():
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_async_main())
