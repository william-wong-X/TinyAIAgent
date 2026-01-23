from __future__ import annotations

from typing import List, Sequence, Union, Iterator, Optional, Set, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path
import hashlib
import sqlite3
import time

SUPPORTED_EXTS = {
    ".pdf",
    ".txt", ".md",
    ".html", ".htm",
    ".docx", ".doc",
    ".pptx", ".ppt",
    ".csv",
    ".json",
}

# ====================== Scan ======================

def iter_files(inputs: Sequence[Union[str, Path]], *, recursive: bool = True) ->Iterator[Path]:
    for item in inputs:
        p = Path(item).expanduser().resolve()
        if p.is_file():
            yield p
        elif p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            for fp in it:
                if fp.is_file():
                    yield fp.resolve()

def iter_supported_files(
    inputs: Sequence[Union[str, Path]], 
    *, 
    recursive: bool = True, 
    only_supported: bool = True
) -> Iterator[Path]:
    for fp in iter_files(inputs, recursive=recursive):
        if only_supported and fp.suffix.lower() not in SUPPORTED_EXTS:
            continue
        yield fp.resolve()

# ====================== Signature ======================

@dataclass
class FileSig:
    size: int
    mtime: int
    sha256: Optional[str] = None

def sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def sig_fast(path: Path) -> FileSig:
    st = path.stat()
    return FileSig(size=int(st.st_size), mtime=int(st.st_mtime), sha256=None)

def sig_strong(path: Path) -> FileSig:
    st = path.stat()
    return FileSig(size=int(st.st_size), mtime=int(st.st_mtime), sha256=sha256_file(path))

# ====================== Change Detector ======================

@dataclass
class ChangeSet:
    added: List[str]
    modified: List[str]
    deleted: List[str]
    unchanged: List[str]

class ManifestDB:
    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path.as_posix())
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                source TEXT PRIMARY KEY, 
                size INTEGER NOT NULL, 
                mtime INTEGER NOT NULL, 
                sha256 TEXT, 
                updated_at INTEGER NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                source TEXT NOT NULL, 
                chunk_id TEXT NOT NULL, 
                PRIMARY KEY (source, chunk_id)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
    
    def get_all_sources(self) -> Set[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT source FROM files")
        return {r[0] for r in cur.fetchall()}
    
    def get_sig(self, source: str) -> Optional[FileSig]:
        cur = self.conn.cursor()
        cur.execute("SELECT size, mtime, sha256 FROM files WHERE source=?", (source,))
        r = cur.fetchone()
        if not r:
            return None
        return FileSig(size=int(r[0]), mtime=int(r[1]), sha256=r[2])
    
    def set_sig(self, source: str, sig: FileSig) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO files(source, size, mtime, sha256, updated_at)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(source) DO UPDATE SET
                size=excluded.size, 
                mtime=excluded.mtime, 
                sha256=excluded.sha256, 
                updated_at=excluded.updated_at
            """,
            (source, sig.size, sig.mtime, sig.sha256, int(time.time())),
        )
        self.conn.commit()

    def delete_source(self, source: str) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM files WHERE source=?", (source,))
        cur.execute("DELETE FROM chunks WHERE source=?", (source,))
        self.conn.commit()

    def get_chunk_ids(self, source: str) -> List[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT chunk_id FROM chunks WHERE source=?", (source,))
        return [r[0] for r in cur.fetchall()]

    def replace_chunk_ids(self, source: str, chunk_ids: List[str]) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM chunks WHERE source=?", (source,))
        cur.executemany(
            "INSERT OR IGNORE INTO chunks(source, chunk_id) VALUES(?, ?)",
            [(source, cid) for cid in chunk_ids],
        )
        self.conn.commit()

class ChangeDetector:
    def __init__(
        self, 
        *, 
        manifest: ManifestDB, 
        recursive: bool = True, 
        only_supported: bool = True, 
        confirm_by_sha256_on_suspect: bool = True
    ) -> None:
        self.manifest = manifest
        self.recursive = recursive
        self.only_supported = only_supported
        self.confirm_by_sha256_on_suspect = confirm_by_sha256_on_suspect

    def detect(self, kb_root: Union[str, Path]) -> Tuple[ChangeSet, Dict[str, FileSig]]:
        prev_sources = self.manifest.get_all_sources()

        current_sources: Set[str] = set()
        current_sigs: Dict[str, FileSig] = {}

        for fp in iter_supported_files([kb_root], recursive=self.recursive, only_supported=self.only_supported):
            source = str(fp.resolve())
            current_sources.add(source)
            current_sigs[source] = sig_fast(fp)
        
        added = sorted(list(current_sources - prev_sources))
        deleted = sorted(list(prev_sources - current_sources))

        unchanged: List[str] = []
        suspects: List[str] = []
        modified: List[str] = []

        for source in sorted(list(current_sources & prev_sources)):
            old = self.manifest.get_sig(source)
            if old is None:
                added.append(source)
                continue
            new_fast = current_sigs[source]
            if (new_fast.size, new_fast.mtime) == (old.size, old.mtime):
                unchanged.append(source)
            else:
                suspects.append(source)

        if self.confirm_by_sha256_on_suspect and suspects:
            for source in suspects:
                new_strong = sig_strong(Path(source))
                current_sigs[source] = new_strong
                old = self.manifest.get_sig(source)
                if old and old.sha256 and old.sha256 == new_strong.sha256:
                    unchanged.append(source)
                else:
                    modified.append(source)
        else:
            modified.extend(suspects)

        return ChangeSet(added=added, modified=modified, deleted=deleted, unchanged=unchanged), current_sigs