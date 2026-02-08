#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import shutil
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import requests
import chromadb

# Optional (recommended) for fast cosine similarity
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


# ============================================================
# Regex / Config
# ============================================================

DOC_RE = re.compile(r"^\s*БАРИМТ\s*[:.]\s*(.+?)\s*$", re.IGNORECASE)

# Chapters
CHAPTER_COLON_RE = re.compile(
    r"^\s*(НЭГ|ХОЁР|ГУРАВ|ДӨРӨВ|ТАВ|ЗУРГАА|ДОЛОО|НАЙМ|ЕС|АРАВ|АРВАН\s+НЭГ|АРВАН\s+ХОЁР)\s*[:.]\s*(.+?)\s*$",
    re.IGNORECASE
)
CHAPTER_BULEG_RE = re.compile(
    r"^\s*(НЭГДҮГЭЭР|ХОЁРДУГААР|ГУРАВДУГААР|ДӨРӨВДҮГЭЭР|ТАВДУГААР|ЗУРГАДУГААР|ДОЛООДУГААР|НАЙМДУГААР|ЕСДҮГЭЭР|АРАВДУГААР|АРВАН\s+НЭГДҮГЭЭР|АРВАН\s+ХОЁРДУГААР)\s+БҮЛЭГ\s*[:.]\s*(.*)\s*$",
    re.IGNORECASE
)
APPENDIX_RE = re.compile(
    r"^\s*ХАВСРАЛТ\s*([0-9]+(?:\s*-\s*[0-9]+)?)?\s*[:.]\s*(.*)\s*$",
    re.IGNORECASE
)

# Clause patterns (indexing)
CLAUSE_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:\.)?\s*(.*)\s*$")
CLAUSE_ALT_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:[)\-–—])\s*(.*)\s*$")

# RAW clause start pattern (extraction)
RAW_CLAUSE_START_RE = re.compile(
    r"^\s*(?:[-•\u2022]\s*)?(\d+(?:\.\d+)*)(?:\s*[.)\-–—]|\.)\s*(.*)\s*$"
)

# Avoid treating year lines as clauses
YEAR_LINE_RE = re.compile(r"^\s*(19\d{2}|20\d{2})\s+он(ы|)\b", re.IGNORECASE)

# Tokenizer
TOKEN_RE = re.compile(r"[А-ЯӨҮЁа-яөүё0-9]+", re.UNICODE)

# KB Q/A markers
KB_Q_RE = re.compile(r"^\s*Асуулт\s*[:：]\s*(.*)\s*$", re.IGNORECASE)
KB_A_RE = re.compile(r"^\s*Хариулт\s*[:：]\s*(.*)\s*$", re.IGNORECASE)


# ============================================================
# Helpers
# ============================================================

def read_text_with_fallback(fp: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "utf-16", "cp1251", "cp866"):
        try:
            return fp.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return fp.read_text(encoding="utf-8", errors="ignore")


def normalize_space(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


def tokens(s: str) -> List[str]:
    return TOKEN_RE.findall(s.lower())


def lexical_overlap_score(q: str, doc: str) -> int:
    qset = set(tokens(q))
    if not qset:
        return 0
    dset = set(tokens(doc))
    return len(qset & dset)


def looks_like_clause_start(line: str) -> Optional[Tuple[str, str]]:
    if YEAR_LINE_RE.match(line):
        return None
    m = CLAUSE_RE.match(line)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m2 = CLAUSE_ALT_RE.match(line)
    if m2:
        return m2.group(1).strip(), m2.group(2).strip()
    return None


@dataclass
class ChapterCtx:
    chapter_id: str
    chapter_title: str


def detect_doc_header(line: str) -> Optional[str]:
    m = DOC_RE.match(line)
    if not m:
        return None
    return normalize_space(m.group(1))


def detect_chapter(line: str) -> Optional[ChapterCtx]:
    m = APPENDIX_RE.match(line)
    if m:
        num = normalize_space(m.group(1) or "")
        title = normalize_space(m.group(2) or "")
        chap_id = f"ХАВСРАЛТ {num}".strip() if num else "ХАВСРАЛТ"
        return ChapterCtx(chapter_id=chap_id, chapter_title=title or chap_id)

    m = CHAPTER_BULEG_RE.match(line)
    if m:
        ordinal = normalize_space(m.group(1))
        title = normalize_space(m.group(2) or "")
        chap_id = f"{ordinal} БҮЛЭГ"
        return ChapterCtx(chapter_id=chap_id, chapter_title=title or chap_id)

    m = CHAPTER_COLON_RE.match(line)
    if m:
        chap = normalize_space(m.group(1))
        title = normalize_space(m.group(2) or "")
        return ChapterCtx(chapter_id=chap, chapter_title=title or chap)

    return None


def finalize_text(lines: List[str]) -> str:
    cleaned = []
    for ln in lines:
        x = ln.rstrip()
        if x.strip():
            cleaned.append(x.strip())
    return "\n".join(cleaned).strip()


def make_chunk_id(source_file: str, chapter_id: str, chunk_type: str, clause_id: Optional[str], idx: int) -> str:
    if chunk_type == "clause":
        return f"{source_file}::[{chapter_id}]::clause::{clause_id}"
    return f"{source_file}::[{chapter_id}]::preamble::{idx}"


# ============================================================
# Chunking for Chroma build
# ============================================================

def parse_txt_to_chunks(text: str, source_file: str) -> List[Dict[str, Any]]:
    """
    Indexing chunker.
    Important: final LLM context is built from RAW extraction, not these chunk bodies.
    """
    lines = text.splitlines()

    doc_title: str = ""
    chapter: Optional[ChapterCtx] = None

    preamble_lines: List[str] = []
    in_clause = False
    clause_id: Optional[str] = None
    clause_lines: List[str] = []

    preamble_count = 0
    chunks: List[Dict[str, Any]] = []

    def flush_preamble():
        nonlocal preamble_lines, preamble_count
        if chapter and preamble_lines:
            body = finalize_text(preamble_lines)
            if body:
                preamble_count += 1
                chunks.append({
                    "id": make_chunk_id(source_file, chapter.chapter_id, "preamble", None, preamble_count),
                    "chunk_type": "preamble",
                    "source_file": source_file,
                    "doc_title": doc_title,
                    "chapter_id": chapter.chapter_id,
                    "chapter_title": chapter.chapter_title,
                    "clause_id": "",
                    "text": body,
                    "is_definition": False,
                    "definition_terms": "",
                })
        preamble_lines = []

    def flush_clause():
        nonlocal in_clause, clause_id, clause_lines
        if in_clause and chapter and clause_id:
            body = finalize_text(clause_lines)
            if body:
                chunks.append({
                    "id": make_chunk_id(source_file, chapter.chapter_id, "clause", clause_id, 0),
                    "chunk_type": "clause",
                    "source_file": source_file,
                    "doc_title": doc_title,
                    "chapter_id": chapter.chapter_id,
                    "chapter_title": chapter.chapter_title,
                    "clause_id": clause_id,
                    "text": body,
                    "is_definition": False,
                    "definition_terms": "",
                })
        in_clause = False
        clause_id = None
        clause_lines = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        dh = detect_doc_header(line)
        if dh is not None:
            flush_clause()
            flush_preamble()
            doc_title = dh
            continue

        ch = detect_chapter(line)
        if ch is not None:
            flush_clause()
            flush_preamble()
            chapter = ch
            preamble_count = 0
            continue

        cs = looks_like_clause_start(line)
        if cs:
            if chapter is None:
                chapter = ChapterCtx(chapter_id="ҮНДСЭН", chapter_title="ҮНДСЭН")
                preamble_count = 0

            flush_clause()
            flush_preamble()
            cid, remainder = cs
            in_clause = True
            clause_id = cid
            clause_lines = []
            if remainder:
                clause_lines.append(remainder)
            continue

        if chapter is None:
            continue

        if in_clause:
            clause_lines.append(raw.rstrip())
        else:
            preamble_lines.append(raw.rstrip())

    flush_clause()
    flush_preamble()
    return chunks


def _dedupe_ids(chunks: List[Dict[str, Any]]) -> None:
    seen: Dict[str, int] = {}
    for c in chunks:
        base = c["id"]
        if base not in seen:
            seen[base] = 1
            continue
        seen[base] += 1
        c["id"] = f"{base}#{seen[base]}"


# ============================================================
# Ollama HTTP client with timing metrics
# ============================================================

@dataclass
class LLMMetrics:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_eval_time: float = 0.0 
    generation_time: float = 0.0 
    total_time: float = 0.0     
    tokens_per_sec: float = 0.0

    def __str__(self) -> str:
        return (
            f"\n[LLM METRICS]\n"
            f"  Prompt tokens: {self.prompt_tokens}\n"
            f"  Completion tokens: {self.completion_tokens}\n"
            f"  Prompt eval time: {self.prompt_eval_time:.3f}s\n"
            f"  Generation time: {self.generation_time:.3f}s\n"
            f"  Total LLM time: {self.total_time:.3f}s\n"
            f"  Tokens/sec: {self.tokens_per_sec:.1f}\n"
        )


class OllamaHTTP:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def embed_texts(self, texts: List[str], model: str) -> List[List[float]]:
        # batch
        try:
            data = self._post("/api/embed", {"model": model, "input": texts})
            embs = data.get("embeddings")
            if isinstance(embs, list) and embs and isinstance(embs[0], list):
                return embs
        except Exception:
            pass

        # fallback per text
        out: List[List[float]] = []
        for t in texts:
            data = self._post("/api/embeddings", {"model": model, "prompt": t})
            emb = data.get("embedding")
            if not emb:
                raise RuntimeError("Ollama embeddings returned no embedding.")
            out.append(emb)
        return out

    def chat(self, system: str, user: str, model: str, options: Dict[str, Any]) -> Tuple[str, LLMMetrics]:
        """
        Returns (response_text, metrics)
        """
        start_time = time.time()
        
        data = self._post("/api/chat", {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": options or {}
        })
        
        end_time = time.time()
        total_time = end_time - start_time
        
        msg = data.get("message", {}) or {}
        response = (msg.get("content") or "").strip()
        
        # Extract metrics from Ollama response
        metrics = LLMMetrics()
        
        # Token counts
        metrics.prompt_tokens = data.get("prompt_eval_count", 0)
        metrics.completion_tokens = data.get("eval_count", 0)
        
        # Times (Ollama returns in nanoseconds, convert to seconds)
        prompt_eval_duration = data.get("prompt_eval_duration", 0)
        eval_duration = data.get("eval_duration", 0)
        
        metrics.prompt_eval_time = prompt_eval_duration / 1e9 if prompt_eval_duration else 0.0
        metrics.generation_time = eval_duration / 1e9 if eval_duration else 0.0
        metrics.total_time = total_time
        
        # Tokens per second
        if metrics.generation_time > 0 and metrics.completion_tokens > 0:
            metrics.tokens_per_sec = metrics.completion_tokens / metrics.generation_time
        
        return response, metrics


# ============================================================
# RAW clause extraction cache (robust section grabbing)
# ============================================================

def _clause_level(cid: str) -> int:
    return len(cid.split(".")) if cid else 0


def _is_descendant(parent: str, child: str) -> bool:
    return bool(parent and child and child.startswith(parent + "."))


def _is_real_clause_id_heuristic(cid: str) -> bool:
    # Avoid list items like "1." inside clauses
    if not cid:
        return False
    if "." in cid:
        return True
    try:
        return int(cid) >= 10
    except Exception:
        return False


class RawDocCache:
    def __init__(self, docs_dir: Path):
        self.docs_dir = docs_dir
        self._lines_cache: Dict[str, List[str]] = {}
        self._chapters_cache: Dict[str, Dict[str, Tuple[int, int]]] = {}
        self._marks_cache: Dict[Tuple[str, str], List[Tuple[int, str, int]]] = {}
        self._startmap_cache: Dict[Tuple[str, str], Dict[str, int]] = {}

    def _load_lines(self, source_file: str) -> List[str]:
        if source_file in self._lines_cache:
            return self._lines_cache[source_file]
        fp = self.docs_dir / source_file
        if not fp.exists():
            self._lines_cache[source_file] = []
            return []
        text = read_text_with_fallback(fp)
        lines = text.splitlines()
        self._lines_cache[source_file] = lines
        return lines

    def _build_chapter_bounds(self, source_file: str) -> Dict[str, Tuple[int, int]]:
        if source_file in self._chapters_cache:
            return self._chapters_cache[source_file]

        lines = self._load_lines(source_file)
        bounds: Dict[str, Tuple[int, int]] = {}

        starts: List[Tuple[str, int]] = []
        for i, raw in enumerate(lines):
            s = raw.strip()
            if not s:
                continue
            ch = detect_chapter(s)
            if ch:
                starts.append((ch.chapter_id, i))

        if not starts:
            # implicit single chapter
            bounds["ҮНДСЭН"] = (0, len(lines))
        else:
            for idx, (cid, sidx) in enumerate(starts):
                eidx = starts[idx + 1][1] if idx + 1 < len(starts) else len(lines)
                bounds[cid] = (sidx, eidx)

        self._chapters_cache[source_file] = bounds
        return bounds

    def _build_clause_marks(self, source_file: str, chapter_id: str) -> Tuple[List[Tuple[int, str, int]], Dict[str, int]]:
        key = (source_file, chapter_id)
        if key in self._marks_cache and key in self._startmap_cache:
            return self._marks_cache[key], self._startmap_cache[key]

        lines = self._load_lines(source_file)
        if not lines:
            self._marks_cache[key] = []
            self._startmap_cache[key] = {}
            return [], {}

        chap_bounds = self._build_chapter_bounds(source_file)
        if chapter_id not in chap_bounds:
            chap_start, chap_end = 0, len(lines)
        else:
            chap_start, chap_end = chap_bounds[chapter_id]

        marks: List[Tuple[int, str, int]] = []
        startmap: Dict[str, int] = {}

        for i in range(chap_start, chap_end):
            s = lines[i].strip()
            if not s:
                continue
            if YEAR_LINE_RE.match(s):
                continue
            m = RAW_CLAUSE_START_RE.match(s)
            if not m:
                continue
            cid = (m.group(1) or "").strip()
            if not _is_real_clause_id_heuristic(cid):
                continue
            lvl = _clause_level(cid)
            marks.append((i, cid, lvl))
            # first occurrence only
            startmap.setdefault(cid, i)

        marks.sort(key=lambda x: x[0])
        self._marks_cache[key] = marks
        self._startmap_cache[key] = startmap
        return marks, startmap

    def extract_clause_section(self, source_file: str, chapter_id: str, clause_id: str, max_lines: int = 600) -> str:
        lines = self._load_lines(source_file)
        if not lines:
            return ""

        chap_bounds = self._build_chapter_bounds(source_file)
        if chapter_id not in chap_bounds:
            chapter_id = "ҮНДСЭН"

        marks, startmap = self._build_clause_marks(source_file, chapter_id)
        start_idx = startmap.get(clause_id)

        # fallback: try scan whole chapter for exact start pattern
        if start_idx is None:
            chap_start, chap_end = chap_bounds.get(chapter_id, (0, len(lines)))
            esc = re.escape(clause_id)
            start_re = re.compile(rf"^\s*(?:[-•\u2022]\s*)?{esc}(?:\s*[.)\-–—]|\.)\s*.*$")
            for i in range(chap_start, chap_end):
                s = lines[i].strip()
                if YEAR_LINE_RE.match(s):
                    continue
                if start_re.match(s):
                    start_idx = i
                    break

        if start_idx is None:
            return ""

        chap_start, chap_end = chap_bounds.get(chapter_id, (0, len(lines)))
        target_level = _clause_level(clause_id)

        end_idx = chap_end
        # walk forward in marks after start_idx and stop at first sibling/ancestor boundary
        for (i, cid, lvl) in marks:
            if i <= start_idx:
                continue
            if cid == clause_id:
                continue
            if _is_descendant(clause_id, cid):
                continue
            # end at same/higher level (<= target_level)
            if target_level >= 2:
                if lvl <= target_level and lvl >= 2:
                    end_idx = i
                    break
            else:
                # for level-1 clauses, any next mark is boundary
                end_idx = i
                break

        section_lines: List[str] = []
        for i in range(start_idx, min(end_idx, chap_end)):
            section_lines.append(lines[i].rstrip())
            if len(section_lines) >= max_lines:
                break

        return "\n".join(section_lines).strip()


# ============================================================
# KB Q/A loader + cosine similarity index
# ============================================================

@dataclass
class KBEntry:
    source_file: str
    question: str
    answer: str


def parse_kb_qa_file(fp: Path) -> List[KBEntry]:
    text = read_text_with_fallback(fp)
    lines = text.splitlines()

    entries: List[KBEntry] = []

    q_lines: List[str] = []
    a_lines: List[str] = []
    state = "idle"  # idle | q | a

    def flush():
        nonlocal q_lines, a_lines, state
        q = normalize_space(" ".join([x.strip() for x in q_lines if x.strip()]))
        a = finalize_text(a_lines)
        if q and a:
            entries.append(KBEntry(source_file=fp.name, question=q, answer=a))
        q_lines = []
        a_lines = []
        state = "idle"

    for raw in lines:
        line = raw.rstrip("\n")

        mq = KB_Q_RE.match(line)
        if mq:
            # new question starts => flush previous block
            flush()
            state = "q"
            rest = (mq.group(1) or "").strip()
            if rest:
                q_lines.append(rest)
            continue

        ma = KB_A_RE.match(line)
        if ma:
            state = "a"
            rest = (ma.group(1) or "").strip()
            if rest:
                a_lines.append(rest)
            continue

        # continuation
        if state == "q":
            if line.strip():
                q_lines.append(line.strip())
        elif state == "a":
            a_lines.append(line.rstrip())
        else:
            continue

    flush()
    return entries


def parse_kb_qa_dir(kb_dir: Path) -> List[KBEntry]:
    if not kb_dir.exists():
        return []
    out: List[KBEntry] = []
    for fp in sorted(kb_dir.glob("*.txt")):
        out.extend(parse_kb_qa_file(fp))
    return out


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


class KBQAIndex:
    def __init__(self, entries: List[KBEntry], embed_model: str, cache_dir: Optional[Path] = None):
        self.entries = entries
        self.embed_model = embed_model
        self.cache_dir = cache_dir
        self.emb = None  # normalized embeddings (np.ndarray)

    def _signature(self) -> str:
        blob = self.embed_model + "\n"
        for e in self.entries:
            blob += f"{e.source_file}\n{e.question}\n{_sha1(e.answer)}\n"
        return _sha1(blob)

    def _cache_paths(self) -> Tuple[Path, Path]:
        assert self.cache_dir is not None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        meta_fp = self.cache_dir / "kb_index_meta.json"
        emb_fp = self.cache_dir / "kb_index_emb.npy"
        return meta_fp, emb_fp

    def load_or_build(self, ollama: OllamaHTTP, batch_size: int = 64) -> None:
        if np is None:
            raise RuntimeError("numpy is required for KB cosine similarity. Install: pip install numpy")

        if not self.entries:
            self.emb = np.zeros((0, 1), dtype=np.float32)
            return

        sig = self._signature()

        # try load cache
        if self.cache_dir:
            meta_fp, emb_fp = self._cache_paths()
            if meta_fp.exists() and emb_fp.exists():
                try:
                    meta = json.loads(meta_fp.read_text(encoding="utf-8"))
                    if meta.get("signature") == sig and meta.get("embed_model") == self.embed_model:
                        self.emb = np.load(str(emb_fp))
                        return
                except Exception:
                    pass

        # build embeddings
        questions = [e.question for e in self.entries]
        vecs: List[List[float]] = []
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            vecs.extend(ollama.embed_texts(batch, model=self.embed_model))

        mat = np.asarray(vecs, dtype=np.float32)
        # normalize
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        mat = mat / norms
        self.emb = mat

        # save cache
        if self.cache_dir:
            meta_fp, emb_fp = self._cache_paths()
            meta_fp.write_text(json.dumps({
                "signature": sig,
                "embed_model": self.embed_model,
                "count": len(self.entries),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            np.save(str(emb_fp), self.emb)

    def search(self, q_emb: List[float], top_n: int = 2, min_sim: float = 0.80) -> List[Tuple[KBEntry, float]]:
        if np is None:
            return []
        if self.emb is None or len(self.entries) == 0:
            return []
        qv = np.asarray(q_emb, dtype=np.float32)
        qv = qv / (np.linalg.norm(qv) + 1e-12)
        sims = (self.emb @ qv).astype(np.float32)

        if top_n <= 0:
            return []

        idx = np.argpartition(-sims, min(top_n, len(sims)) - 1)[:min(top_n, len(sims))]
        idx = idx[np.argsort(-sims[idx])]

        out: List[Tuple[KBEntry, float]] = []
        for i in idx:
            s = float(sims[i])
            if s >= min_sim:
                out.append((self.entries[int(i)], s))
        return out


# ============================================================
# Index build (Chroma)
# ============================================================

def build_index(
    input_dir: Path,
    db_dir: Path,
    collection_name: str,
    embed_model: str,
    out_jsonl: Optional[Path],
    reset: bool,
    ollama_url: str,
    batch_size: int = 64
) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in: {input_dir}")

    all_chunks: List[Dict[str, Any]] = []
    for fp in txt_files:
        all_chunks.extend(parse_txt_to_chunks(read_text_with_fallback(fp), source_file=fp.name))

    _dedupe_ids(all_chunks)

    if out_jsonl:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as f:
            for c in all_chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_dir))

    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    ollama = OllamaHTTP(base_url=ollama_url)

    ids = [c["id"] for c in all_chunks]
    docs = [c["text"] for c in all_chunks]

    metas: List[Dict[str, Any]] = []
    for c in all_chunks:
        metas.append({
            "source_file": c["source_file"],
            "doc_title": c["doc_title"],
            "chapter_id": c["chapter_id"],
            "chapter_title": c["chapter_title"],
            "chunk_type": c["chunk_type"],
            "clause_id": c["clause_id"] or "",
            "is_definition": bool(c.get("is_definition", False)),
            "definition_terms": c.get("definition_terms", ""),
        })

    print(f"[BUILD] Files: {len(txt_files)}")
    print(f"[BUILD] Total chunks: {len(ids)}")
    print(f"[BUILD] Chroma DB: {db_dir}")
    print(f"[BUILD] Collection: {collection_name}")
    print(f"[BUILD] Embed model: {embed_model}")
    print(f"[BUILD] Ollama URL: {ollama_url}")

    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_docs = docs[i:i+batch_size]
        batch_meta = metas[i:i+batch_size]
        embs = ollama.embed_texts(batch_docs, model=embed_model)
        collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_meta, embeddings=embs)
        print(f"[BUILD] Upserted {min(i+batch_size, len(ids))}/{len(ids)}")

    print("[BUILD] Done.")


# ============================================================
# Chat: retrieval + raw extraction + KB injection
# ============================================================

def rerank_and_filter(
    q: str,
    candidates: List[Tuple[str, Dict[str, Any], float]],
    top_k: int,
    dist_threshold: float
) -> List[Tuple[str, Dict[str, Any], float]]:
    kept = [(d, m, dist) for (d, m, dist) in candidates if dist is not None and dist <= dist_threshold]
    if not kept:
        return []

    overlapped = [(d, m, dist) for (d, m, dist) in kept if lexical_overlap_score(q, d) >= 1]
    if overlapped:
        kept = overlapped

    kept.sort(key=lambda x: (lexical_overlap_score(q, x[0]), -x[2]), reverse=True)
    return kept[:top_k]


def build_prompts(q: str, context: str) -> Tuple[str, str]:
    system_prompt = (
        "Та МУИС-ийн журам, дүрмийн ишлэлүүд дээр суурилсан туслах.\n"
        "ЗОРИЛГО: Хүнд ойлгомжтой, ярианы өнгө аяс бага зэрэг шингэсэн, байгалийн монгол хэлээр тайлбарлан хариул.\n"
        "\n"
        "ХАТУУ ДҮРЭМ:\n"
        "- ЗӨВХӨН SOURCES-д буй мэдээллээр хариул. Таамаг, нэмэлт мэдлэг, ерөнхий тайлбар нэмж болохгүй.\n"
        "- SOURCES-д байхгүй бол 'SOURCES-д энэ талаар мэдээлэл олдсонгүй.' гэж товч бич.\n"
        "- Хариултаа ЗӨВХӨН кирилл монголоор бич.\n"
        "- Латин үсгийг ЗӨВХӨН 'Эх сурвалж:' хэсэгт (NUMxxxx.txt гэх мэт) ашиглаж болно.\n"
        "Хангасан бүх баримтыг уншиж зөв баримтыг ашиглан асуултанд хариул\n"
        "\n"
        "ЗАГВАР:\n"
        "1) Хариултыг 2–5 богино догол мөр эсвэл жагсаалтаар өг.\n"
        "2) Ишлэлийг ОЛОН ДАВТАЖ өгөлгүй: төгсгөлд нь 'Эх сурвалж:' хэсэгт ашигласан эх сурвалжуудыг жагсаа.\n"
        "   - Журам/дүрмийн ишлэл бол: file|chapter|clause\n"
        "   - KB (асуулт-хариулт) ишлэл бол: KB|file|QA\n"
    )
    user_prompt = f"SOURCES:\n{context}\n\nАСУУЛТ:\n{q}\n\nХАРИУЛТ:\n"
    return system_prompt, user_prompt


def build_context_from_raw(
    kept: List[Tuple[str, Dict[str, Any], float]],
    raw_cache: RawDocCache,
    max_chars: int
) -> str:
    blocks: List[str] = []
    total = 0
    seen = set()

    for doc, meta, _ in kept:
        sf = (meta.get("source_file") or "").strip()
        ch = (meta.get("chapter_id") or "").strip() or "ҮНДСЭН"
        cid = (meta.get("clause_id") or "").strip()
        ctype = (meta.get("chunk_type") or "").strip()

        key = (sf, ch, cid, ctype)
        if key in seen:
            continue
        seen.add(key)

        text = ""
        if ctype == "clause" and sf and cid:
            text = raw_cache.extract_clause_section(sf, ch, cid)
            if not text:
                text = (doc or "").strip()
        else:
            text = (doc or "").strip()

        if not text:
            continue

        src = f"{sf}|{ch}|{cid}".rstrip("|")
        block = f"[{src}]\n{text}\n"

        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 300:
                blocks.append(block[:remaining])
            break

        blocks.append(block)
        total += len(block)

    return "\n".join(blocks).strip()


def build_context_with_kb(
    base_context: str,
    kb_hits: List[Tuple[KBEntry, float]],
    max_chars: int
) -> str:
    if not kb_hits:
        return base_context

    blocks = [base_context] if base_context else []
    used = len(base_context)

    blocks.append("\n===== KB МЭДЭЭЛЭЛ =====\n")
    used += len(blocks[-1])

    for e, sim in kb_hits:
        # IMPORTANT: do NOT include literal "Асуулт:" / "Хариулт:" markers
        kb_block = (
            f"[KB|{e.source_file}|QA]\n"
            f"Сонгогдсон асуултын агуулга: {e.question}\n"
            f"Тохирох мэдээлэл:\n{e.answer}\n"
        )
        if used + len(kb_block) > max_chars:
            break
        blocks.append(kb_block)
        used += len(kb_block)

    return "\n".join(blocks).strip()


def chat_loop(
    db_dir: Path,
    collection_name: str,
    llm_model: str,
    embed_model: str,
    top_k: int,
    dist_threshold: float,
    num_ctx: int,
    num_gpu: int,
    num_thread: int,
    num_predict: int,
    ollama_url: str,
    debug: bool,
    max_context_chars: int,
    docs_dir: Path,
    kb_dir: Path,
    kb_top_n: int,
    kb_min_sim: float,
    kb_cache_dir: Optional[Path],
) -> None:
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    ollama = OllamaHTTP(base_url=ollama_url)
    raw_cache = RawDocCache(docs_dir=docs_dir)

    # Load KB
    kb_entries = parse_kb_qa_dir(kb_dir)
    kb_index = KBQAIndex(entries=kb_entries, embed_model=embed_model, cache_dir=kb_cache_dir)
    if kb_entries:
        kb_index.load_or_build(ollama)

    opts: Dict[str, Any] = {
        "num_ctx": num_ctx,
        "num_gpu": num_gpu,
        "num_thread": num_thread,
        "num_predict": num_predict,
        "temperature": 0.2,
        "top_p": 0.9,
        "repeat_penalty": 1.12,
    }

    print("[CHAT] Type 'exit' to quit.")
    while True:
        try:
            q = input("\nAsk (or 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[CHAT] Bye.")
            return

        if not q:
            continue
        if q.lower() in ("exit", "quit", "q"):
            print("[CHAT] Bye.")
            return

        q_emb = ollama.embed_texts([q], model=embed_model)[0]

        # Retrieve candidates from Chroma
        n_candidates = max(top_k * 8, 32)
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"]
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        candidates = list(zip(docs, metas, dists))

        kept = rerank_and_filter(q=q, candidates=candidates, top_k=top_k, dist_threshold=dist_threshold)

        if debug:
            print("\n[DEBUG] Retrieved (after rerank/filter):")
            for doc, meta, dist in kept:
                print(f"  - dist={dist:.4f} file={meta.get('source_file')} chapter={meta.get('chapter_id')} clause={meta.get('clause_id')} type={meta.get('chunk_type')}")

        if not kept:
            print("SOURCES-д энэ талаар мэдээлэл олдсонгүй.")
            continue

        # Build base context from RAW extraction
        base_context = build_context_from_raw(kept, raw_cache, max_chars=max_context_chars)

        # KB similarity hits
        kb_hits: List[Tuple[KBEntry, float]] = []
        if kb_entries:
            kb_hits = kb_index.search(q_emb, top_n=kb_top_n, min_sim=kb_min_sim)

        if debug and kb_hits:
            print("\n[DEBUG] KB hits:")
            for e, sim in kb_hits:
                print(f"  - sim={sim:.4f} file={e.source_file} q={e.question[:80]}...")

        # Merge KB into context (respect max chars)
        context = build_context_with_kb(base_context, kb_hits, max_chars=max_context_chars)

        system_prompt, user_prompt = build_prompts(q=q, context=context)

        if debug:
            print("\n" + "=" * 80)
            print("[DEBUG] LLM MODEL:", llm_model)
            print("[DEBUG] LLM OPTIONS:", opts)
            print("-" * 80)
            print("[DEBUG] SYSTEM PROMPT BEGIN")
            print(system_prompt)
            print("[DEBUG] SYSTEM PROMPT END")
            print("-" * 80)
            print("[DEBUG] USER PROMPT BEGIN")
            print(user_prompt)
            print("[DEBUG] USER PROMPT END")
            print("=" * 80 + "\n")

        # Call LLM and get metrics
        ans, metrics = ollama.chat(system=system_prompt, user=user_prompt, model=llm_model, options=opts)
        
        if not ans.strip():
            print("Одоогоор загвар хоосон хариу буцаалаа.")
            continue
        
        # Display answer
        print(ans)
        
        # Display metrics
        print(metrics)


# ============================================================
# cleanbak utilities
# ============================================================

def delete_bak_files(folder: Path) -> None:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(str(folder))
    removed = 0
    for fp in folder.glob("*.bak"):
        fp.unlink(missing_ok=True)
        removed += 1
    print(f"[CLEAN] Deleted {removed} .bak files from {folder}")


def move_bak_files(src_folder: Path, dst_folder: Path) -> None:
    src_folder = Path(src_folder)
    dst_folder = Path(dst_folder)
    dst_folder.mkdir(parents=True, exist_ok=True)
    moved = 0
    for fp in src_folder.glob("*.bak"):
        shutil.move(str(fp), str(dst_folder / fp.name))
        moved += 1
    print(f"[CLEAN] Moved {moved} .bak files from {src_folder} -> {dst_folder}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Student Handbook RAG (chunk -> Chroma) + Chat (Ollama)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Build
    p_build = sub.add_parser("build", help="Parse txt files -> chunks -> Chroma")
    p_build.add_argument("--input_dir", required=True)
    p_build.add_argument("--db_dir", required=True)
    p_build.add_argument("--collection", default="student_handbook")
    p_build.add_argument("--embed_model", required=True)
    p_build.add_argument("--out_jsonl", default="")
    p_build.add_argument("--reset", action="store_true")
    p_build.add_argument("--ollama_url", default="http://localhost:11434")
    p_build.add_argument("--batch_size", type=int, default=64)

    # Chat
    p_chat = sub.add_parser("chat", help="Interactive RAG chat")
    p_chat.add_argument("--db_dir", required=True)
    p_chat.add_argument("--collection", default="student_handbook")
    p_chat.add_argument("--llm_model", required=True)
    p_chat.add_argument("--embed_model", required=True)
    p_chat.add_argument("--top_k", type=int, default=4)
    p_chat.add_argument("--dist_threshold", type=float, default=0.48)
    p_chat.add_argument("--num_ctx", type=int, default=16384)
    p_chat.add_argument("--num_gpu", type=int, default=-1)
    p_chat.add_argument("--num_thread", type=int, default=8)
    p_chat.add_argument("--num_predict", type=int, default=1024)
    p_chat.add_argument("--ollama_url", default="http://localhost:11434")
    p_chat.add_argument("--debug", action="store_true")
    p_chat.add_argument("--max_context_chars", type=int, default=14000)

    p_chat.add_argument("--docs_dir", default=r"C:\Student_handbook_supporter\clean_txt")

    # KB options
    p_chat.add_argument("--kb_dir", default=r"C:\Student_handbook_supporter\kb\sentences")
    p_chat.add_argument("--kb_top_n", type=int, default=2)
    p_chat.add_argument("--kb_min_sim", type=float, default=0.80)
    p_chat.add_argument("--kb_cache_dir", default=r"C:\Student_handbook_supporter\kb\_cache")

    # Clean .bak
    p_clean = sub.add_parser("cleanbak", help="Delete or move *.bak files")
    p_clean.add_argument("--dir", required=True)
    p_clean.add_argument("--move_to", default="")

    args = parser.parse_args()

    if args.cmd == "build":
        out_jsonl = Path(args.out_jsonl) if args.out_jsonl else None
        build_index(
            input_dir=Path(args.input_dir),
            db_dir=Path(args.db_dir),
            collection_name=args.collection,
            embed_model=args.embed_model,
            out_jsonl=out_jsonl,
            reset=args.reset,
            ollama_url=args.ollama_url,
            batch_size=args.batch_size
        )
        return

    if args.cmd == "chat":
        kb_cache = Path(args.kb_cache_dir) if args.kb_cache_dir else None
        chat_loop(
            db_dir=Path(args.db_dir),
            collection_name=args.collection,
            llm_model=args.llm_model,
            embed_model=args.embed_model,
            top_k=args.top_k,
            dist_threshold=args.dist_threshold,
            num_ctx=args.num_ctx,
            num_gpu=args.num_gpu,
            num_thread=args.num_thread,
            num_predict=args.num_predict,
            ollama_url=args.ollama_url,
            debug=args.debug,
            max_context_chars=args.max_context_chars,
            docs_dir=Path(args.docs_dir),
            kb_dir=Path(args.kb_dir),
            kb_top_n=args.kb_top_n,
            kb_min_sim=args.kb_min_sim,
            kb_cache_dir=kb_cache,
        )
        return

    if args.cmd == "cleanbak":
        folder = Path(args.dir)
        if args.move_to:
            move_bak_files(folder, Path(args.move_to))
        else:
            delete_bak_files(folder)
        return


if __name__ == "__main__":
    main()