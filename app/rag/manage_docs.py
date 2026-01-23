from __future__ import annotations

from typing import List, Optional
import argparse

from app.model_client import create_embedding
from app.rag.chromaDB import ChromaVectorStore, create_vector_store
from app.rag.doc_process import preprocess
from app.rag.docs_change_detector import ManifestDB, ChangeDetector, FileSig
from config.config import PreprocessConfig, SyncConfig, load_config

def sync_once(
        config: SyncConfig, 
        *, 
        chroma: ChromaVectorStore, 
        pp_config: Optional[PreprocessConfig] = None
) -> None:
    pp_config = pp_config or PreprocessConfig()
    db = ManifestDB(config.manifest_path)
    try:
        detector = ChangeDetector(
            manifest=db, 
            recursive=pp_config.loader.recursive, 
            only_supported=pp_config.loader.only_supported, 
            confirm_by_sha256_on_suspect=config.confirm_by_sha256_on_suspect
        )
        changes, current_sigs = detector.detect(config.kb_root)
        print(f"[SCAN] added={len(changes.added)} modified={len(changes.modified)} deleted={len(changes.deleted)} unchanged={len(changes.unchanged)}")

        # deleted
        for source in changes.deleted:
            old_ids = db.get_chunk_ids(source)
            print(f"[DEL] {source} chunks={len(old_ids)}")
            if not config.dry_run and old_ids:
                chroma.delete(old_ids)
            if not config.dry_run:
                db.delete_source(source)
        
        # modified
        for source in changes.modified:
            old_ids = db.get_chunk_ids(source)
            print(f"[MOD] {source} old_chunks={len(old_ids)} -> rebuild")
            if not config.dry_run and old_ids:
                chroma.delete(old_ids)
            if not config.dry_run:
                _build_and_upsert_one(
                    source=source, 
                    sig=current_sigs[source], 
                    pp_config=pp_config, 
                    chroma=chroma, 
                    manifest=db, 
                    embed_batch=config.embed_batch
                )
        
        # added
        for source in changes.added:
            print(f"[ADD] {source} -> build")
            if not config.dry_run:
                _build_and_upsert_one(
                    source=source, 
                    sig=current_sigs[source], 
                    pp_config=pp_config,  
                    chroma=chroma, 
                    manifest=db, 
                    embed_batch=config.embed_batch
                )

        print("[DONE] sync finished.")

    finally:
        db.close()

def _build_and_upsert_one(
    *, 
    source: str, 
    sig: FileSig, 
    pp_config: PreprocessConfig, 
    chroma: ChromaVectorStore, 
    manifest: ManifestDB, 
    embed_batch: int
) -> None:
    chunks = list(preprocess([source], config=pp_config))

    if not chunks:
        manifest.set_sig(source, sig)
        manifest.replace_chunk_ids(source, [])
        return 
    
    ids = [c.id for c in chunks]
    texts = [c.text for c in chunks]
    metas = [dict(c.metadata or {}) for c in chunks]

    inserted: List[str] = []
    try:
        for i in range(0, len(texts), embed_batch):
            t_batch = texts[i:i + embed_batch]
            id_batch = ids[i:i + embed_batch]
            m_batch = metas[i:i + embed_batch]

            returned_ids = chroma.upsert_texts(
                texts=t_batch,
                metadatas=m_batch,
                ids=id_batch,
            )
            inserted.extend(returned_ids)
        manifest.set_sig(source, sig)
        manifest.replace_chunk_ids(source, ids)
    except Exception:
        try:
            if inserted:
                chroma.delete(ids=inserted)
        except Exception:
            pass
        raise

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-c', '--config', type=str, default="./config/config.yaml", help="Config path")

    args = parser.parse_args()

    config = load_config(args.config)

    sync_cfg = config.sync
    pp_config = config.preprocess
    vdb_cfg = config.vectordb.chroma

    embedding_fn = create_embedding(config).embed_documents
    chroma = create_vector_store(vdb_cfg, embedding_fn)

    sync_once(
        sync_cfg,
        chroma=chroma,
        pp_config=pp_config,
    )

    try:
        chroma.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()