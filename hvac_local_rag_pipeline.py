"""
HVAC Local RAG Benchmark Pipeline - R19
========================================
Mục đích:
  - Xây dựng RAG cục bộ từ tài liệu PDF/DOCX của bạn (QCVN, ASHRAE, TCVN...).
  - So sánh LLM thuần (Gemini) vs Local RAG (mô phỏng HIP).
  - Xuất CSV log minh chứng thực chứng cho bài báo.

Không cần HIP API — chỉ cần thư mục chứa các file PDF/DOCX tài liệu kỹ thuật.

Cài đặt:
  pip install google-generativeai chromadb pypdf2 python-docx pandas numpy

Cách chạy:
  1. Đặt GEMINI_API_KEY bên dưới.
  2. Đặt DOCS_FOLDER trỏ tới thư mục chứa file PDF/DOCX.
  3. python hvac_local_rag_pipeline.py
"""

import asyncio
import io
import json
import os
import re
import sys
import time
import uuid
import numpy as np
import pandas as pd

# Fix encoding cho Windows terminal (tranh loi cp1252 voi tieng Viet)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ─── CẤU HÌNH ───────────────────────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyBP8yIlAfxWEV8EOrbwBG0B2f1otiqlCfs"

# Thu muc chua tai lieu tieu chuan ky thuat (6 PDF: QCVN, ASHRAE, TCVN)
DOCS_FOLDER = r"C:\TONG HOP\DATA\DHCN\VIET BAO AI FOR HVAC\ASHRARE 2026\BAI 01\TIEU CHUAN"

NUM_SAMPLES       = 1200    # Bo du lieu day du cho bai bao
OUTPUT_CSV        = r"C:\TONG HOP\DATA\DHCN\VIET BAO AI FOR HVAC\THU KY NCKH\HVAC_Benchmark_N1200_RawLogs.csv"
CHECKPOINT_CSV    = r"C:\TONG HOP\DATA\DHCN\VIET BAO AI FOR HVAC\THU KY NCKH\HVAC_Benchmark_N1200_Checkpoint.csv"
CHROMA_DIR        = r"C:\TONG HOP\DATA\DHCN\VIET BAO AI FOR HVAC\ASHRARE 2026\BAI 01\TIEU CHUAN\.chroma_db"
SLEEP_BETWEEN_CALLS = 4.5  # Giay nghi giua 2 cap API (gioi han ~13 req/phut < 15 req/phut free tier)
MAX_RETRIES       = 3       # So lan thu lai neu bi rate-limit (429)

# Ground Truth theo QCVN / ASHRAE (dùng để chấm điểm tự động)
GROUND_TRUTH = {
    "Ventilation":     {"value": 20.0,  "unit": "m3/h/person", "standard": "TCVN 5687:2024"},
    "Chiller_COP":     {"value": 6.1,   "unit": "COP",         "standard": "QCVN 09:2017"},
    "LPD":             {"value": 8.5,   "unit": "W/m2",        "standard": "LEED v4.1"},
    "U_Factor":        {"value": 0.36,  "unit": "W/m2.K",      "standard": "ASHRAE 90.1"},
    "Thermal_Comfort": {"value": 27.0,  "unit": "degC",        "standard": "ASHRAE 55 + QCVN 26:2016"},
    "Air_Filter":      {"value": 13,    "unit": "MERV",        "standard": "LEED v4.1 + TCVN 13521:2022"},
}
TOLERANCE = {
    "Ventilation": 0.5, "Chiller_COP": 0.05, "LPD": 0.2,
    "U_Factor": 0.02,   "Thermal_Comfort": 0.5, "Air_Filter": 0,
}


# ─── MODULE 1: ĐỌC TÀI LIỆU ─────────────────────────────────────────────────
def load_documents(folder: str) -> list[dict]:
    """Đọc tất cả PDF, DOCX, TXT trong thư mục (đệ quy).
    Trả về list[{"filename": str, "text": str}]
    """
    import glob

    docs = []

    # --- PDF ---
    for path in glob.glob(os.path.join(folder, "**", "*.pdf"), recursive=True):
        try:
            import PyPDF2
            text = ""
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += (page.extract_text() or "") + "\n"
            if text.strip():
                docs.append({"filename": os.path.basename(path), "text": text})
                print(f"  ✅ PDF: {os.path.basename(path)} ({len(text):,} ký tự)")
        except Exception as e:
            print(f"  ⚠️  Bỏ qua {path}: {e}")

    # --- DOCX ---
    for path in glob.glob(os.path.join(folder, "**", "*.docx"), recursive=True):
        try:
            import docx as _docx
            d = _docx.Document(path)
            text = "\n".join(p.text for p in d.paragraphs)
            if text.strip():
                docs.append({"filename": os.path.basename(path), "text": text})
                print(f"  ✅ DOCX: {os.path.basename(path)} ({len(text):,} ký tự)")
        except Exception as e:
            print(f"  ⚠️  Bỏ qua {path}: {e}")

    # --- TXT ---
    for path in glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True):
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
            if text.strip():
                docs.append({"filename": os.path.basename(path), "text": text})
                print(f"  ✅ TXT: {os.path.basename(path)} ({len(text):,} ký tự)")
        except Exception as e:
            print(f"  ⚠️  Bỏ qua {path}: {e}")

    return docs


# ─── MODULE 2: XÂY DỰNG VECTOR DATABASE (ChromaDB) ──────────────────────────
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Chia văn bản thành các đoạn nhỏ có độ chồng lấp."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def build_or_load_vectordb(docs: list[dict], persist_dir: str):
    """Xay dung ChromaDB tu tai lieu (hoac load lai neu da build)."""
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=persist_dir)

    # Thu Gemini embedding, neu loi thi dung SentenceTransformer offline
    ef = None
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)

        # Kiem tra model hop le bang cach thu embed 1 cau ngan
        test_emb = genai.embed_content(
            model="models/gemini-embedding-exp-03-07",
            content="test",
            task_type="retrieval_document"
        )

        class GeminiEmbedding(embedding_functions.EmbeddingFunction):
            def __call__(self, input: list[str]) -> list[list[float]]:
                result = []
                for text in input:
                    emb = genai.embed_content(
                        model="models/gemini-embedding-exp-03-07",
                        content=text[:2000],  # Gioi han do dai
                        task_type="retrieval_document"
                    )
                    result.append(emb["embedding"])
                return result

        ef = GeminiEmbedding()
        print("  -> Su dung Gemini Embedding (gemini-embedding-exp-03-07)")
    except Exception as e:
        print(f"  -> Gemini Embedding loi ({e.__class__.__name__}), dung SentenceTransformer offline...")

    if ef is None:
        try:
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="paraphrase-multilingual-MiniLM-L12-v2"
            )
            print("  -> SentenceTransformer (paraphrase-multilingual-MiniLM-L12-v2)")
        except Exception as e2:
            # Fallback cuoi cung: dung embedding mac dinh cua ChromaDB
            print(f"  -> Dung ChromaDB default embedding ({e2})")
            ef = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name="hvac_knowledge_base",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    # Nếu collection rỗng → build mới
    if collection.count() == 0:
        print(f"  → Đang index {len(docs)} tài liệu vào ChromaDB...")
        all_chunks, all_ids, all_metas = [], [], []
        for doc in docs:
            chunks = chunk_text(doc["text"])
            for j, chunk in enumerate(chunks):
                cid = f"{doc['filename']}_{j}"
                all_chunks.append(chunk)
                all_ids.append(cid)
                all_metas.append({"source": doc["filename"]})

        # Upsert theo batch 100 để tránh lỗi size
        batch = 100
        for start in range(0, len(all_chunks), batch):
            collection.upsert(
                documents=all_chunks[start:start+batch],
                ids=all_ids[start:start+batch],
                metadatas=all_metas[start:start+batch]
            )
        print(f"  → Đã index {collection.count()} đoạn văn bản.")
    else:
        print(f"  → Load ChromaDB đã có sẵn ({collection.count()} đoạn).")

    return collection


# ─── MODULE 3: QUERY RAG ─────────────────────────────────────────────────────
def query_local_rag(collection, prompt: str, top_k: int = 5) -> tuple[str, float]:
    """Truy xuất top_k đoạn liên quan → tổng hợp câu trả lời bằng Gemini."""
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)

    start = time.monotonic()

    # Retrieval
    results = collection.query(query_texts=[prompt], n_results=top_k)
    retrieved_chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    context = "\n\n---\n\n".join(retrieved_chunks)

    # Generation với ngữ cảnh từ tài liệu thực
    augmented_prompt = f"""Bạn là kỹ sư thẩm định HVAC. Chỉ sử dụng thông tin trong [TÀI LIỆU KỸ THUẬT] dưới đây để trả lời.
TUYỆT ĐỐI không bịa thêm tiêu chuẩn. Nếu không tìm thấy thông tin, ghi rõ "Không tìm thấy trong tài liệu".

[TÀI LIỆU KỸ THUẬT]
{context}

[CÂU HỎI]
{prompt}

Trả lời DẠNG JSON:
{{
  "standard_applied": "tên tiêu chuẩn trích từ tài liệu",
  "calculated_value": <số>,
  "unit": "đơn vị",
  "compliance_status": "Pass hoặc Fail",
  "source_files": {json.dumps(list(set(sources)), ensure_ascii=False)}
}}"""

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        augmented_prompt,
        request_options={"timeout": 90}
    )
    latency = round(time.monotonic() - start, 3)

    return response.text.strip(), latency


# ─── MODULE 4: GỌI GEMINI THUẦN (LLM Baseline) ───────────────────────────────
def query_llm_baseline(prompt: str) -> tuple[str, float]:
    """Gọi Gemini không có RAG context — đây là LLM phổ thông."""
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)

    start = time.monotonic()
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        prompt,
        request_options={"timeout": 90}
    )
    latency = round(time.monotonic() - start, 3)
    return response.text.strip(), latency


# ─── MODULE 5: TRÍCH XUẤT & CHẤM ĐIỂM ───────────────────────────────────────
def extract_value(text: str, category: str) -> dict:
    """JSON parse → RegEx fallback."""
    # --- JSON ---
    try:
        m = re.search(r'\{.*?\}', text, re.DOTALL)
        if m:
            data = json.loads(m.group())
            val = data.get("calculated_value", -1)
            return {
                "extracted_value":   float(val) if val is not None else -1,
                "extracted_standard": str(data.get("standard_applied", "")),
                "compliance_status": str(data.get("compliance_status", "")),
                "parse_method": "JSON"
            }
    except (json.JSONDecodeError, ValueError):
        pass

    # --- RegEx ---
    PATTERNS = {
        "Ventilation":     r'(\d+\.?\d*)\s*m.?3.*?(?:h|giờ).*?người',
        "Chiller_COP":     r'COP\s*(?:tối thiểu|≥|>=|:)?\s*(\d+\.?\d*)',
        "LPD":             r'(?:LPD|mật độ)\D{0,20}(\d+\.?\d*)\s*W',
        "U_Factor":        r'U\D{0,20}(\d+\.?\d*)\s*W',
        "Thermal_Comfort": r'(\d{2,3}\.?\d*)\s*°?C',
        "Air_Filter":      r'MERV\s*(\d+)',
    }
    pat = PATTERNS.get(category, r'(\d+\.?\d*)')
    m2 = re.search(pat, text, re.IGNORECASE)
    return {
        "extracted_value":   float(m2.group(1)) if m2 else -1,
        "extracted_standard": "",
        "compliance_status": "",
        "parse_method": "RegEx" if m2 else "Failed"
    }


def auto_grade(category: str, extracted: dict) -> dict:
    gt     = GROUND_TRUTH[category]
    gt_val = gt["value"]
    gt_std = gt["standard"].lower()
    ex_val = extracted["extracted_value"]
    ex_std = extracted["extracted_standard"].lower()
    tol    = TOLERANCE[category]

    std_ok = any(s.strip() in ex_std for s in gt_std.split("+"))
    val_ok = (abs(ex_val - gt_val) <= tol) if ex_val != -1 else False

    if not std_ok:
        etype = "Lỗi Không gian Pháp lý"
    elif not val_ok:
        etype = "Lỗi Nội suy Đo lường"
    else:
        etype = "None"

    is_hallu = (extracted["parse_method"] != "Failed") and (not std_ok or not val_ok)
    return {
        "is_correct": std_ok and val_ok,
        "is_hallucination": is_hallu,
        "error_typology": etype,
        "gt_value": gt_val,
        "gt_standard": gt["standard"]
    }


# ─── MODULE 6: SINH DỮ LIỆU ─────────────────────────────────────────────────
def generate_hvac_testbed(num_samples: int) -> pd.DataFrame:
    rng   = np.random.default_rng(seed=42)
    n     = num_samples // 6
    rows  = []

    specs = [
        ("Ventilation",
         lambda: {"area": round(float(rng.uniform(50, 1500)), 1),
                  "occ":  int(np.ceil(float(rng.uniform(50, 1500)) / float(rng.uniform(8, 15))))},
         lambda p: (
             f"Tính gió tươi cho văn phòng {p['area']}m², {p['occ']} người, TP.HCM. "
             "Áp dụng tiêu chuẩn hiện hành. "
             "Trả lời JSON: {\"standard_applied\": \"...\", \"calculated_value\": <số>, "
             "\"unit\": \"m3/h/person\", \"compliance_status\": \"Pass/Fail\"}."
         )),
        ("Chiller_COP",
         lambda: {"cop": round(float(rng.uniform(5.0, 7.0)), 2)},
         lambda p: (
             f"Chiller ly tâm 300 TR, COP={p['cop']}. Đạt QCVN 09:2017 không? "
             "JSON: {\"standard_applied\": \"...\", \"calculated_value\": <COP min>, "
             "\"unit\": \"COP\", \"compliance_status\": \"Pass/Fail\"}."
         )),
        ("LPD",
         lambda: {"lpd": round(float(rng.uniform(6.0, 12.0)), 1)},
         lambda p: (
             f"LPD = {p['lpd']} W/m². Có đạt LEED v4.1 không? "
             "JSON: {\"standard_applied\": \"...\", \"calculated_value\": <giới hạn>, "
             "\"unit\": \"W/m2\", \"compliance_status\": \"Pass/Fail\"}."
         )),
        ("U_Factor",
         lambda: {"u": round(float(rng.uniform(0.2, 0.8)), 2)},
         lambda p: (
             f"Vỏ bao che U={p['u']} W/m².K tại TP.HCM. Đạt ASHRAE 90.1 không? "
             "JSON: {\"standard_applied\": \"...\", \"calculated_value\": <U max>, "
             "\"unit\": \"W/m2.K\", \"compliance_status\": \"Pass/Fail\"}."
         )),
        ("Thermal_Comfort",
         lambda: {"temp": round(float(rng.normal(26, 2)), 1)},
         lambda p: (
             f"Văn phòng {p['temp']}°C Hà Nội, có quạt HVLS v=0.8m/s. Đạt ASHRAE 55? "
             "JSON: {\"standard_applied\": \"...\", \"calculated_value\": <nhiệt độ>, "
             "\"unit\": \"degC\", \"compliance_status\": \"Pass/Fail\"}."
         )),
        ("Air_Filter",
         lambda: {"merv": int(rng.choice([8, 10, 11, 13, 14, 16]))},
         lambda p: (
             f"AHU dùng màng lọc MERV {p['merv']}. Đạt LEED v4.1 + TCVN 13521:2022? "
             "JSON: {\"standard_applied\": \"...\", \"calculated_value\": <MERV min>, "
             "\"unit\": \"MERV\", \"compliance_status\": \"Pass/Fail\"}."
         )),
    ]

    for category, gen_params, build_prompt in specs:
        for _ in range(n):
            params = gen_params()
            rows.append({
                "Test_ID":  f"TC-{uuid.uuid4().hex[:8].upper()}",
                "Category": category,
                "Params":   json.dumps(params, ensure_ascii=False),
                "Prompt":   build_prompt(params),
                "Expected": GROUND_TRUTH[category]["standard"]
            })

    return pd.DataFrame(rows)


# === MODULE 7: PIPELINE CHINH (Checkpoint + Retry + Rate-limit safe) ===
# Cot dau ra duoc sap xep theo nhom: ID / Input / LLM / HIP / GroundTruth
ORDERED_COLS = [
    # --- Nhan dien ---
    "No", "Test_ID", "Category", "Timestamp",
    # --- Dau vao ---
    "Params", "Raw_Prompt",
    # --- LLM Baseline ---
    "LLM_Status", "LLM_Latency_sec", "LLM_ParseMethod",
    "LLM_Extracted_Val", "LLM_Extracted_Std",
    "LLM_IsCorrect", "LLM_IsHallucination", "LLM_ErrorTypology",
    "LLM_Response",
    # --- HIP (Local RAG) ---
    "HIP_Status", "HIP_Latency_sec", "HIP_ParseMethod",
    "HIP_Extracted_Val", "HIP_Extracted_Std",
    "HIP_IsCorrect",
    "HIP_Response",
    # --- Ground Truth ---
    "GT_Value", "GT_Standard",
]


def _call_with_retry(fn, *args):
    """Goi ham API voi hard-timeout 120s dung concurrent.futures.
    Neu bi rate-limit (429) thi sleep 65s roi thu lai.
    Neu timeout hoac loi khac thi tra ve ERROR string."""
    import concurrent.futures
    import google.api_core.exceptions as gex

    for attempt in range(MAX_RETRIES):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(fn, *args)
                try:
                    return future.result(timeout=120)   # Hard-kill sau 120s
                except concurrent.futures.TimeoutError:
                    print(f"    [TIMEOUT] API khong tra loi sau 120s, bo qua dong nay.")
                    return "ERROR: Timeout 120s", 0.0
        except gex.ResourceExhausted:
            wait = 65 * (attempt + 1)
            print(f"    [RETRY] Rate-limit 429, cho {wait}s (lan {attempt+1}/{MAX_RETRIES})...")
            time.sleep(wait)
        except Exception as e:
            return f"ERROR: {e}", 0.0
    return "ERROR: Max retries exceeded", 0.0


def run_pipeline(df: pd.DataFrame, collection) -> pd.DataFrame:
    records = []
    total   = len(df)

    # Load checkpoint neu co
    done_ids = set()
    if os.path.exists(CHECKPOINT_CSV):
        df_ck = pd.read_csv(CHECKPOINT_CSV, encoding="utf-8-sig")
        done_ids = set(df_ck["Test_ID"].tolist())
        records   = df_ck.to_dict("records")
        print(f"  -> Resume tu checkpoint: da co {len(done_ids)} dong.")

    for idx, row in df.iterrows():
        if row["Test_ID"] in done_ids:
            continue

        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  [{idx+1:04d}/{total}] {row['Category']} ...")

        prompt   = row["Prompt"]
        category = row["Category"]

        # --- LLM Baseline ---
        result_llm = _call_with_retry(query_llm_baseline, prompt)
        if isinstance(result_llm, tuple):
            llm_text, llm_lat = result_llm
            llm_status = "Success" if not llm_text.startswith("ERROR") else "Failed"
        else:
            llm_text, llm_lat, llm_status = str(result_llm), 0.0, "Failed"

        llm_ext   = extract_value(llm_text, category)
        llm_grade = auto_grade(category, llm_ext)

        # --- Local RAG ---
        result_rag = _call_with_retry(query_local_rag, collection, prompt)
        if isinstance(result_rag, tuple):
            rag_text, rag_lat = result_rag
            rag_status = "Success" if not rag_text.startswith("ERROR") else "Failed"
        else:
            rag_text, rag_lat, rag_status = str(result_rag), 0.0, "Failed"

        rag_ext   = extract_value(rag_text, category)
        rag_grade = auto_grade(category, rag_ext)

        rec = {
            "No":                  idx + 1,
            "Test_ID":             row["Test_ID"],
            "Category":            category,
            "Timestamp":           pd.Timestamp.now().isoformat(timespec="seconds"),
            "Params":              row["Params"],
            "Raw_Prompt":          prompt[:280],
            "LLM_Status":          llm_status,
            "LLM_Latency_sec":     round(llm_lat, 3),
            "LLM_ParseMethod":     llm_ext["parse_method"],
            "LLM_Extracted_Val":   llm_ext["extracted_value"],
            "LLM_Extracted_Std":   llm_ext["extracted_standard"],
            "LLM_IsCorrect":       llm_grade["is_correct"],
            "LLM_IsHallucination": llm_grade["is_hallucination"],
            "LLM_ErrorTypology":   llm_grade["error_typology"],
            "LLM_Response":        llm_text[:380],
            "HIP_Status":          rag_status,
            "HIP_Latency_sec":     round(rag_lat, 3),
            "HIP_ParseMethod":     rag_ext["parse_method"],
            "HIP_Extracted_Val":   rag_ext["extracted_value"],
            "HIP_Extracted_Std":   rag_ext["extracted_standard"],
            "HIP_IsCorrect":       rag_grade["is_correct"],
            "HIP_Response":        rag_text[:380],
            "GT_Value":            llm_grade["gt_value"],
            "GT_Standard":         llm_grade["gt_standard"],
        }
        records.append(rec)

        # Luu checkpoint moi 50 dong
        if len(records) % 50 == 0:
            df_ck = pd.DataFrame(records)[ORDERED_COLS]
            df_ck.to_csv(CHECKPOINT_CSV, index=False, encoding="utf-8-sig")
            print(f"    [CHECKPOINT] Da luu {len(records)}/{total} dong.")

        # Rate-limit safe: ngu SLEEP_BETWEEN_CALLS giay
        time.sleep(SLEEP_BETWEEN_CALLS)

    df_out = pd.DataFrame(records)
    # Dam bao thu tu cot
    cols = [c for c in ORDERED_COLS if c in df_out.columns]
    return df_out[cols]


# ─── MODULE 8: THỐNG KÊ ──────────────────────────────────────────────────────
def print_statistics(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("THỐNG KÊ ĐỊNH LƯỢNG (N={})".format(len(df)))
    print("=" * 60)
    n = len(df)
    stats = {
        "LLM_Accuracy_%":          round(df["LLM_IsCorrect"].mean() * 100, 2),
        "LLM_Accuracy_SD_%":       round(df["LLM_IsCorrect"].std() * 100, 2),
        "LLM_Hallucination_%":     round(df["LLM_IsHallucination"].mean() * 100, 2),
        "LLM_Latency_mean_sec":    round(df["LLM_Latency_sec"].mean(), 3),
        "LLM_Latency_SD_sec":      round(df["LLM_Latency_sec"].std(), 3),
        "HIP_Accuracy_%":          round(df["HIP_IsCorrect"].mean() * 100, 2),
        "HIP_Latency_mean_sec":    round(df["HIP_Latency_sec"].mean(), 3),
        "HIP_Latency_SD_sec":      round(df["HIP_Latency_sec"].std(), 3),
        "Error_Legal_%":           round((df["LLM_ErrorTypology"] == "Lỗi Không gian Pháp lý").sum() / n * 100, 2),
        "Error_Measurement_%":     round((df["LLM_ErrorTypology"] == "Lỗi Nội suy Đo lường").sum() / n * 100, 2),
        "Error_None_%":            round((df["LLM_ErrorTypology"] == "None").sum() / n * 100, 2),
    }
    for k, v in stats.items():
        print(f"  {k:<35s}: {v}")

    # Xuất CSV thống kê
    stats_csv = OUTPUT_CSV.replace(".csv", "_Statistics.csv")
    pd.DataFrame([stats]).to_csv(stats_csv, index=False, encoding="utf-8-sig")
    print(f"\n  → Lưu thống kê: {stats_csv}")
    return stats


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("HVAC Local RAG Benchmark Pipeline - R19")
    print("=" * 60)

    # 1. Đọc tài liệu training
    print("\n[1/5] Đọc tài liệu từ:", DOCS_FOLDER)
    docs = load_documents(DOCS_FOLDER)
    if not docs:
        print("  ⚠️  Không tìm thấy tài liệu nào! Kiểm tra DOCS_FOLDER.")
        return
    print(f"  → Tổng: {len(docs)} tài liệu")

    # 2. Xây dựng Vector DB
    print("\n[2/5] Xây dựng ChromaDB (vector database)...")
    collection = build_or_load_vectordb(docs, CHROMA_DIR)

    # 3. Sinh bộ dữ liệu kiểm thử
    print(f"\n[3/5] Sinh Testbed (N={NUM_SAMPLES})...")
    df_test = generate_hvac_testbed(NUM_SAMPLES)
    input_csv = OUTPUT_CSV.replace(".csv", "_Input.csv")
    df_test.to_csv(input_csv, index=False, encoding="utf-8-sig")
    print(f"  → Đã sinh {len(df_test)} prompts. Lưu: {input_csv}")

    # 4. Chạy Benchmark
    print("\n[4/5] Chạy Benchmark (LLM Baseline vs Local RAG)...")
    print("  ⏱  Thời gian ước tính: ~{:.0f} phút".format(len(df_test) * 5 / 60))
    df_results = run_pipeline(df_test, collection)

    # 5. Xuất log & thống kê
    print("\n[5/5] Xuất kết quả...")
    df_results.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"  → Raw logs: {OUTPUT_CSV}")
    print_statistics(df_results)

    print("\n✅ Pipeline hoàn tất!")
    print("📎 Đính kèm các file CSV này vào phần Supplementary Data của bài báo.")


if __name__ == "__main__":
    main()
