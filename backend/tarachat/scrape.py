"""Scrape the web for documents."""

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path

import aiofiles
import aiohttp
from bs4 import BeautifulSoup
from yarl import URL

logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT = 60
DEFAULT_CHUNK_SIZE = 1024 * 1024

# Characters forbidden in filenames on Windows and/or Linux.
_FILENAME_BAD_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# Most filesystems cap filename length at 255 bytes. Reserve room for the
# ".meta.json" sidecar suffix (10 bytes) that meta_path_for() appends.
_MAX_FILENAME_BYTES = 255 - len(".meta.json")


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
    if not remote_meta:
        return True
    return (
        remote_meta["etag"] != local_meta.get("etag")
        or remote_meta["last_modified"] != local_meta.get("last_modified")
        or remote_meta["content_length"] != local_meta.get("content_length")
    )


def sanitize_filename(text: str, extension: str = "") -> str:
    """Turn *text* into a safe filename, preserving non-ASCII characters.

    Long filenames are truncated to fit within filesystem limits (255 bytes
    minus room for the ``.meta.json`` sidecar).  A short hash is appended
    when truncation occurs so that distinct original names stay distinct.

    >>> sanitize_filename("Règlement numéro 06.06.2025", ".pdf")
    'Règlement numéro 06.06.2025.pdf'
    >>> sanitize_filename('a/b:c*d', '.txt')
    'a_b_c_d.txt'
    """
    name = _FILENAME_BAD_RE.sub("_", text).strip().strip(".")
    if not name:
        name = "_"
    # Ensure the proper extension is present.
    if extension and not name.lower().endswith(extension.lower()):
        name += extension

    # Truncate if the UTF-8 encoded filename exceeds the byte budget.
    if len(name.encode("utf-8")) > _MAX_FILENAME_BYTES:
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
        suffix = f"_{digest}{extension}"
        suffix_bytes = len(suffix.encode("utf-8"))
        budget = _MAX_FILENAME_BYTES - suffix_bytes

        # Strip the original extension before truncating the stem.
        stem = name[: -len(extension)] if extension else name
        encoded = stem.encode("utf-8")
        truncated = encoded[:budget].decode("utf-8", errors="ignore")
        name = truncated.rstrip() + suffix

    return name


class Downloader:
    """Downloads files, skipping unchanged ones based on HTTP metadata.

    Subclass and override :meth:`fetch_metadata` and :meth:`fetch_content`
    for testing without real network or disk I/O.
    """

    async def fetch_metadata(self, session: aiohttp.ClientSession, url: URL) -> dict:
        """Get metadata via HEAD. If it fails, return ``{}``."""
        try:
            async with session.head(url, allow_redirects=True, timeout=20) as resp:
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

    async def fetch_content(
        self,
        session: aiohttp.ClientSession,
        url: URL,
        file_path: Path,
        *,
        timeout: int = DEFAULT_TIMEOUT,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        """Download *url* and stream the response body to *file_path*."""
        async with session.get(url, timeout=timeout) as resp:
            resp.raise_for_status()
            async with aiofiles.open(file_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(chunk_size):
                    if chunk:
                        await f.write(chunk)

    async def download_one(
        self,
        session: aiohttp.ClientSession,
        url: URL,
        target_dir: Path,
        filename: str | None = None,
        *,
        timeout: int = DEFAULT_TIMEOUT,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> tuple[str, str | None, str]:
        """Download a single URL if it has changed.

        *filename* overrides the default (``url.name``) for the local file.
        """
        filename = filename or url.name
        file_path = target_dir / filename

        logger.info(f"[{url}] Checking remote metadata...")
        remote_meta = await self.fetch_metadata(session, url)
        local_meta = load_metadata(file_path)

        if file_path.exists() and not has_changed(local_meta, remote_meta):
            logger.info(f"[{url}] Unchanged, skipping.")
            return (url, str(file_path), "skipped")

        logger.info(f"[{url}] Downloading to {file_path}...")
        try:
            await self.fetch_content(
                session, url, file_path, timeout=timeout, chunk_size=chunk_size,
            )
            if remote_meta:
                save_metadata(file_path, remote_meta)
            return (url, str(file_path), "downloaded")
        except Exception:
            logger.exception(f"[{url}] Download failed")
            return (url, None, "error")

    async def download_many(
        self,
        urls: list[tuple[URL, str]],
        target_dir: Path,
        *,
        max_concurrency: int = 5,
        timeout: int = DEFAULT_TIMEOUT,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> list[tuple[str, str | None, str]]:
        """Download multiple URLs concurrently if the remote file changed."""
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        semaphore = asyncio.Semaphore(max_concurrency)

        async with aiohttp.ClientSession() as session:

            async def worker(u: URL, fname: str):
                async with semaphore:
                    return await self.download_one(
                        session, u, target_dir, fname, timeout=timeout, chunk_size=chunk_size,
                    )

            tasks = [asyncio.create_task(worker(u, fn)) for u, fn in urls]
            return await asyncio.gather(*tasks)


async def get_urls(url: URL, timeout: int = DEFAULT_TIMEOUT) -> list[tuple[URL, str]]:
    """Fetch document listing and return ``(download_url, filename)`` pairs.

    The *filename* is derived from the link text (which preserves
    non-ASCII characters such as accents) rather than the URL path,
    since the remote storage may strip those characters from URLs.
    """
    async with aiohttp.ClientSession() as session, session.get(url, timeout=timeout) as resp:
        resp.raise_for_status()
        data = await resp.json()
        html_content = data['contenu']
        soup = BeautifulSoup(html_content, "html.parser")
        results: list[tuple[URL, str]] = []
        for a in soup.find_all("a"):
            href = a.get("href")
            if not href:
                continue
            link_url = URL(href)
            # Derive a proper filename from the link text, falling back to
            # the URL name when no text is available.
            link_text = a.get_text(strip=True)
            ext = Path(link_url.name).suffix or ".pdf"
            filename = (
                sanitize_filename(link_text, ext) if link_text else link_url.name
            )
            results.append((link_url, filename))
        return results


async def _async_main():
    url = URL("https://vplus.modellium.com/api/www.notre-dame-du-laus.ca/structure/detail/reglements?localisation=fr")
    url_filename_pairs = await get_urls(url)
    target_dir = Path("data/documents")
    await Downloader().download_many(url_filename_pairs, target_dir)


def main():
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_async_main())
