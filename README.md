# HVAC Intelligence Platform (HIP) - Empirical Benchmark (ASHRAE 2026)

## 📌 Project Overview
This repository contains the empirical benchmark data and the Python evaluation pipeline used for the research article:
> **"Optimizing HVAC Design Time: An Empirical Benchmark of HIP Platform (RAG) vs LLMs from a Task-Technology Fit Perspective"**
> *(Tối ưu thời gian thiết kế HVAC: Đánh giá thực chứng nền tảng HIP (RAG) và LLMs dưới góc nhìn Task-Technology Fit)*

The objective of this project is to quantitatively measure and compare the performance, accuracy, and hallucination rates of standard Large Language Models (LLMs) against the specialized Retrieval-Augmented Generation (RAG) architecture of the HIP platform in solving complex HVAC engineering compliance tasks (ASHRAE 90.1, ASHRAE 62.1, QCVN 09:2017, TCVN 5687:2024).

## 📊 Dataset (N=900)
The benchmark uses mathematical parametrization to automatically generate a testbed of `N=900` HVAC queries with varying room sizes, temperatures, and equipment capacities.
- **File:** [`HVAC_Benchmark_N1200_Checkpoint.csv`](./HVAC_Benchmark_N1200_Checkpoint.csv)
- **Content:** The CSV file logs every single API request, including:
  - Input parameters (Area, Occupancy, Baseline limits)
  - Raw JSON extraction from LLM vs HIP
  - Latency timestamps (in seconds)
  - Ground Truth comparison and Auto-grading (Pass/Fail)
  - Hallucination and Error Typology categorization

## 🛠 Reproducibility (How to run)
To ensure transparency, peer reviewers and researchers can fully reproduce these findings using their own API keys.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/CaoTrungHau/HIP-Empirical-Benchmark.git
   cd HIP-Empirical-Benchmark
   ```
2. **Install dependencies:**
   ```bash
   pip install pandas numpy chromadb PyPDF2 google-generativeai
   ```
3. **Configure API Key:**
   Replace the `GEMINI_API_KEY` inside the script `hvac_local_rag_pipeline.py` with your own Google Gemini / DeepSeek API key.
4. **Execute Pipeline:**
   ```bash
   python hvac_local_rag_pipeline.py
   ```
   *Note: The script features auto-checkpointing and concurrent threading to manage rate limits.*

## 📈 Key Findings
- **Hallucination Rate:** Traditional LLMs exhibit a 34.5% hallucination rate, predominantly failing at legal boundaries (44.9% of failures).
- **RAG Accuracy:** The HIP platform maintains a 98.8% accuracy by strict vector-search referencing.
- **Latency Trade-off:** RAG implementation adds a latency cost (average 3.8s compared to 1.2s of plain LLM) to achieve zero-tolerance compliance.

## 📄 License
This dataset and code are released under the MIT License for academic and research purposes.

---
*Maintained by Cao Trung Hau - Faculty of Heat and Refrigeration Engineering, Industrial University of Ho Chi Minh City (IUH).*
