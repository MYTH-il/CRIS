"""
NCRB RAG + XGBoost Crime Rate Prediction System
================================================
MTech NLP Non-Major Project Scaffold
File order: run phases top to bottom.

Dependencies:
    pip install pdfplumber langchain langchain-community
                sentence-transformers faiss-cpu chromadb
                xgboost shap scikit-learn pandas spacy
    python -m spacy download en_core_web_sm
"""

import re
import json
import pdfplumber
import pandas as pd
import numpy as np
import faiss
import spacy
import xgboost as xgb
import shap
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

PDF_PATH = "NCRB_STATS.pdf"
VECTOR_STORE_PATH = "ncrb_index"
FEATURE_MATRIX_PATH = "ncrb_features.csv"
MODEL_PATH = "xgboost_crime_model.json"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # fast, good for numeric-heavy text
# Swap to "BAAI/bge-base-en-v1.5" for better accuracy

STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim",
    "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
    "West Bengal", "Delhi", "Jammu & Kashmir", "Ladakh", "Puducherry",
    "Chandigarh", "Andaman & Nicobar Islands", "Dadra & Nagar Haveli", "Lakshadweep"
]

# Features to extract per state and their source tables
FEATURE_SCHEMA = {
    "ipc_crime_rate":        {"table": "1A.1",  "unit": "per_lakh",  "valid": (0, 2000)},
    "sll_crime_rate":        {"table": "1A.2",  "unit": "per_lakh",  "valid": (0, 2000)},
    "violent_crime_rate":    {"table": "1C.1",  "unit": "per_lakh",  "valid": (0, 500)},
    "murder_rate":           {"table": "2A.1",  "unit": "per_lakh",  "valid": (0, 20)},
    "crime_against_women_rate": {"table": "3A.1", "unit": "per_lakh", "valid": (0, 300)},
    "cyber_crime_rate":      {"table": "9A",    "unit": "per_lakh",  "valid": (0, 100)},
    "economic_offence_rate": {"table": "8A",    "unit": "per_lakh",  "valid": (0, 200)},
    "juvenile_crime_rate":   {"table": "5A.1",  "unit": "per_lakh",  "valid": (0, 50)},
    "charge_sheeting_rate":  {"table": "19A",   "unit": "percent",   "valid": (0, 100)},
    "conviction_rate":       {"table": "19A.6", "unit": "percent",   "valid": (0, 100)},
    "acquittal_rate":        {"table": "19A.6", "unit": "percent",   "valid": (0, 100)},
    "police_disposal_rate":  {"table": "17A",   "unit": "percent",   "valid": (0, 100)},
}

TARGET_FEATURE = "ipc_crime_rate"   # what XGBoost predicts


# ─────────────────────────────────────────────
# PHASE 1 — PDF PARSING & CHUNKING
# ─────────────────────────────────────────────

class NCRBParser:
    """
    Parses NCRB PDF into two chunk types:
      - table_chunks: one doc per table row (for numeric extraction)
      - prose_chunks: sentence-split snapshot/summary text (for QA)
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.nlp = spacy.load("en_core_web_sm")

    def extract_table_chunks(self) -> list[dict]:
        """
        Extract tables using pdfplumber. Each row → one structured chunk.
        Prepends column headers to every row to preserve context.
        """
        chunks = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                for table in tables:
                    if not table or len(table) < 2:
                        continue

                    # First non-empty row treated as header
                    headers = [str(h).strip() if h else f"col_{i}"
                               for i, h in enumerate(table[0])]

                    for row in table[1:]:
                        if not any(row):
                            continue
                        # Build key-value string — makes regex extraction trivial
                        row_text = " | ".join(
                            f"{headers[i]}: {str(cell).strip()}"
                            for i, cell in enumerate(row)
                            if cell and str(cell).strip()
                        )
                        # Try to detect state name in the row
                        state = self._detect_state(row_text)
                        chunks.append({
                            "text": row_text,
                            "type": "table_row",
                            "page": page_num,
                            "state": state,
                            "source": f"page_{page_num}",
                        })
        return chunks

    def extract_prose_chunks(self, chunk_size: int = 200,
                              overlap: int = 50) -> list[dict]:
        """
        Extract prose text (Snapshot section, chapter summaries).
        Sentence-splits with token overlap.
        """
        chunks = []
        with pdfplumber.open(self.pdf_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"

        # Use spaCy sentence segmentation
        doc = self.nlp(full_text[:1_000_000])  # cap for memory
        sentences = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 20]

        # Sliding window chunking
        for i in range(0, len(sentences), chunk_size - overlap):
            window = sentences[i: i + chunk_size]
            chunk_text = " ".join(window)
            state = self._detect_state(chunk_text)
            chunks.append({
                "text": chunk_text,
                "type": "prose",
                "state": state,
                "source": f"sentences_{i}_{i+chunk_size}",
            })
        return chunks

    def _detect_state(self, text: str) -> str | None:
        """Simple state name detection from chunk text."""
        for state in STATES:
            if state.lower() in text.lower():
                return state
        return None


# ─────────────────────────────────────────────
# PHASE 2 — VECTOR STORE
# ─────────────────────────────────────────────

class VectorStore:
    """
    Dual FAISS index:
      - table_index: for structured numeric queries
      - prose_index: for semantic / explanatory queries
    """

    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.table_index = None
        self.prose_index = None
        self.table_chunks = []
        self.prose_chunks = []

    def build(self, table_chunks: list[dict], prose_chunks: list[dict]):
        print(f"Embedding {len(table_chunks)} table rows...")
        table_texts = [c["text"] for c in table_chunks]
        table_embeds = self.model.encode(table_texts, show_progress_bar=True,
                                          batch_size=64, normalize_embeddings=True)

        print(f"Embedding {len(prose_chunks)} prose chunks...")
        prose_texts = [c["text"] for c in prose_chunks]
        prose_embeds = self.model.encode(prose_texts, show_progress_bar=True,
                                          batch_size=64, normalize_embeddings=True)

        dim = table_embeds.shape[1]
        self.table_index = faiss.IndexFlatIP(dim)   # inner product = cosine for normalised
        self.table_index.add(table_embeds.astype("float32"))

        self.prose_index = faiss.IndexFlatIP(dim)
        self.prose_index.add(prose_embeds.astype("float32"))

        self.table_chunks = table_chunks
        self.prose_chunks = prose_chunks
        print("Vector store built.")

    def save(self, path: str):
        Path(path).mkdir(exist_ok=True)
        faiss.write_index(self.table_index, f"{path}/table.index")
        faiss.write_index(self.prose_index, f"{path}/prose.index")
        with open(f"{path}/table_chunks.json", "w") as f:
            json.dump(self.table_chunks, f, ensure_ascii=False)
        with open(f"{path}/prose_chunks.json", "w") as f:
            json.dump(self.prose_chunks, f, ensure_ascii=False)

    def load(self, path: str):
        self.table_index = faiss.read_index(f"{path}/table.index")
        self.prose_index = faiss.read_index(f"{path}/prose.index")
        with open(f"{path}/table_chunks.json") as f:
            self.table_chunks = json.load(f)
        with open(f"{path}/prose_chunks.json") as f:
            self.prose_chunks = json.load(f)

    def retrieve_table(self, query: str, state: str = None,
                        k: int = 5) -> list[dict]:
        """Retrieve top-k table chunks. Optionally filter by state."""
        q_embed = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.table_index.search(q_embed.astype("float32"), k * 3)
        results = [self.table_chunks[i] for i in indices[0]]

        if state:
            # Prioritise state-matching chunks
            state_results = [r for r in results if r.get("state") == state]
            other_results = [r for r in results if r.get("state") != state]
            results = (state_results + other_results)[:k]
        return results[:k]

    def retrieve_prose(self, query: str, k: int = 4) -> list[dict]:
        """Retrieve top-k prose chunks for explanatory queries."""
        q_embed = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.prose_index.search(q_embed.astype("float32"), k)
        return [self.prose_chunks[i] for i in indices[0]]


# ─────────────────────────────────────────────
# PHASE 3 — FEATURE EXTRACTION
# ─────────────────────────────────────────────

class FeatureExtractor:
    """
    For each state × feature combination, runs a RAG query
    then deterministically extracts the numeric value via regex.
    LLM is NOT used for number production — only for retrieval routing.
    """

    # Regex patterns per unit type
    PATTERNS = {
        "per_lakh": [
            r"(?:crime rate|rate)[:\s]*([\d.]+)",
            r"([\d.]+)\s*per lakh",
            r":\s*([\d.]+)\s*(?:\||\Z)",
        ],
        "percent": [
            r"([\d.]+)\s*%",
            r"rate[:\s]*([\d.]+)",
            r":\s*([\d.]+)\s*(?:\||\Z)",
        ],
    }

    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store

    def extract_feature(self, state: str, feature_name: str) -> float | None:
        """
        Extract a single numeric feature for a state.
        Returns float or None if extraction fails.
        """
        schema = FEATURE_SCHEMA[feature_name]
        table_ref = schema["table"]
        unit = schema["unit"]
        valid_range = schema["valid"]

        # Construct targeted query
        query = (f"{feature_name.replace('_', ' ')} for {state} "
                 f"Table {table_ref} 2022")

        chunks = self.vs.retrieve_table(query, state=state, k=5)

        for chunk in chunks:
            value = self._extract_number(chunk["text"], unit)
            if value is not None and valid_range[0] <= value <= valid_range[1]:
                return value

        return None   # triggers Tier 2 / Tier 3 fallback

    def _extract_number(self, text: str, unit: str) -> float | None:
        """Deterministic regex extraction — no LLM involved."""
        patterns = self.PATTERNS.get(unit, self.PATTERNS["percent"])
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None

    def build_feature_matrix(self) -> pd.DataFrame:
        """
        Build the full 36 × 12 feature matrix.
        Runs all state × feature combinations sequentially.
        """
        records = []
        for state in STATES:
            row = {"state": state}
            for feature_name in FEATURE_SCHEMA:
                value = self.extract_feature(state, feature_name)
                row[feature_name] = value
                status = f"{value:.1f}" if value else "MISSING"
                print(f"  {state[:15]:15s} | {feature_name[:25]:25s} | {status}")
            records.append(row)

        df = pd.DataFrame(records).set_index("state")

        # Report coverage
        total = df.shape[0] * df.shape[1]
        missing = df.isna().sum().sum()
        print(f"\nExtraction complete. Coverage: {(total-missing)/total*100:.1f}%")
        print(f"Missing values: {missing}/{total}")
        return df


# ─────────────────────────────────────────────
# PHASE 4 — XGBOOST TRAINING
# ─────────────────────────────────────────────

class CrimeRatePredictor:
    """
    XGBoost regressor trained on the extracted feature matrix.
    Predicts IPC crime rate per lakh population.
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.feature_names = None
        self.explainer = None

    def prepare(self, df: pd.DataFrame):
        """Separate features from target, impute, scale."""
        self.feature_names = [c for c in df.columns if c != TARGET_FEATURE]
        X = df[self.feature_names].values
        y = df[TARGET_FEATURE].values

        # Impute missing values (Tier 3 fallback)
        X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)

        return X, y

    def train(self, df: pd.DataFrame):
        X, y = self.prepare(df)

        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="rmse",
        )

        # With 36 states: Leave-One-Out CV is most reliable
        loo = LeaveOneOut()
        cv_scores = cross_val_score(
            self.model, X, y,
            cv=loo,
            scoring="neg_mean_absolute_error"
        )
        print(f"\nLOO-CV MAE: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

        # Fit on full data for deployment
        self.model.fit(X, y)

        # SHAP explainer — key output for your project report
        self.explainer = shap.TreeExplainer(self.model)
        shap_values = self.explainer.shap_values(X)

        print("\nFeature importance (mean |SHAP|):")
        importances = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=self.feature_names
        ).sort_values(ascending=False)
        print(importances.to_string())
        return importances

    def predict(self, state_features: dict) -> dict:
        """
        Predict crime rate for a single state.
        Returns prediction + SHAP explanation.
        """
        row = [state_features.get(f, np.nan) for f in self.feature_names]
        X = self.imputer.transform([row])
        X = self.scaler.transform(X)

        pred = self.model.predict(X)[0]
        shap_vals = self.explainer.shap_values(X)[0]

        explanation = {
            feat: round(float(shap_vals[i]), 3)
            for i, feat in enumerate(self.feature_names)
        }
        top_driver = max(explanation, key=lambda k: abs(explanation[k]))

        return {
            "predicted_crime_rate": round(float(pred), 1),
            "national_avg": 258.1,   # NCRB 2022 figure
            "above_average": float(pred) > 258.1,
            "top_shap_driver": top_driver,
            "shap_explanation": explanation,
        }

    def save(self, path: str):
        self.model.save_model(path)

    def load(self, path: str):
        self.model = xgb.XGBRegressor()
        self.model.load_model(path)
        self.explainer = shap.TreeExplainer(self.model)


# ─────────────────────────────────────────────
# PHASE 5 — ONLINE INFERENCE (UNIFIED PIPELINE)
# ─────────────────────────────────────────────

class NCRBSystem:
    """
    Unified inference: takes a natural language query,
    returns RAG answer + crime rate prediction + SHAP explanation.
    """

    def __init__(self, vector_store: VectorStore,
                 extractor: FeatureExtractor,
                 predictor: CrimeRatePredictor,
                 feature_matrix: pd.DataFrame):
        self.vs = vector_store
        self.extractor = extractor
        self.predictor = predictor
        self.feature_matrix = feature_matrix

    def query(self, user_query: str) -> dict:
        # 1. Detect state in query
        state = self._detect_state(user_query)

        # 2. Determine query type
        is_numeric = self._is_numeric_query(user_query)

        # 3. RAG retrieval
        if is_numeric and state:
            chunks = self.vs.retrieve_table(user_query, state=state)
        else:
            chunks = self.vs.retrieve_prose(user_query)

        rag_context = "\n".join(c["text"] for c in chunks)

        # 4. ML prediction (if state is identifiable)
        prediction = None
        if state and state in self.feature_matrix.index:
            state_features = self.feature_matrix.loc[state].to_dict()
            prediction = self.predictor.predict(state_features)

        return {
            "query": user_query,
            "detected_state": state,
            "rag_context": rag_context[:800],  # truncated for display
            "sources": [c["source"] for c in chunks],
            "prediction": prediction,
        }

    def _detect_state(self, text: str) -> str | None:
        for state in STATES:
            if state.lower() in text.lower():
                return state
        return None

    def _is_numeric_query(self, text: str) -> bool:
        numeric_keywords = ["rate", "number", "count", "how many",
                            "percentage", "%", "statistic", "figure"]
        return any(k in text.lower() for k in numeric_keywords)

    def format_response(self, result: dict) -> str:
        lines = []
        lines.append(f"Query: {result['query']}")
        if result["detected_state"]:
            lines.append(f"State detected: {result['detected_state']}")
        lines.append(f"\nRAG context (top chunks):\n{result['rag_context'][:400]}...")
        if result["prediction"]:
            p = result["prediction"]
            lines.append(f"\nPredicted IPC crime rate: {p['predicted_crime_rate']} per lakh")
            lines.append(f"National avg (2022): {p['national_avg']} per lakh")
            lines.append(f"Above national average: {p['above_average']}")
            lines.append(f"Top SHAP driver: {p['top_shap_driver']} "
                         f"({p['shap_explanation'][p['top_shap_driver']]:+.2f})")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# MAIN — RUN FULL PIPELINE
# ─────────────────────────────────────────────

def build_pipeline():
    """Run once to build the knowledge base and train the model."""

    # Step 1: Parse PDF
    print("=== STEP 1: Parsing PDF ===")
    parser = NCRBParser(PDF_PATH)
    table_chunks = parser.extract_table_chunks()
    prose_chunks = parser.extract_prose_chunks()
    print(f"Extracted {len(table_chunks)} table rows, {len(prose_chunks)} prose chunks")

    # Step 2: Build vector store
    print("\n=== STEP 2: Building vector store ===")
    vs = VectorStore()
    vs.build(table_chunks, prose_chunks)
    vs.save(VECTOR_STORE_PATH)

    # Step 3: Extract feature matrix
    print("\n=== STEP 3: Extracting feature matrix ===")
    extractor = FeatureExtractor(vs)
    feature_matrix = extractor.build_feature_matrix()
    feature_matrix.to_csv(FEATURE_MATRIX_PATH)
    print(f"Feature matrix saved to {FEATURE_MATRIX_PATH}")

    # Step 4: Train XGBoost
    print("\n=== STEP 4: Training XGBoost ===")
    predictor = CrimeRatePredictor()
    importances = predictor.train(feature_matrix)
    predictor.save(MODEL_PATH)

    return vs, extractor, feature_matrix, predictor


def run_demo(vs, extractor, feature_matrix, predictor):
    """Demo queries after pipeline is built."""
    system = NCRBSystem(vs, extractor, predictor, feature_matrix)

    demo_queries = [
        "What is Kerala's conviction rate and how does it compare to national average?",
        "What is Rajasthan's charge-sheeting rate for IPC crimes in 2022?",
        "Why did cyber crime increase in 2022?",
        "Predict crime rate for Uttar Pradesh given its judicial metrics",
    ]

    for query in demo_queries:
        print("\n" + "="*60)
        result = system.query(query)
        print(system.format_response(result))


if __name__ == "__main__":
    import sys

    if "--build" in sys.argv:
        vs, extractor, feature_matrix, predictor = build_pipeline()
        run_demo(vs, extractor, feature_matrix, predictor)
    else:
        # Load pre-built pipeline
        vs = VectorStore()
        vs.load(VECTOR_STORE_PATH)
        feature_matrix = pd.read_csv(FEATURE_MATRIX_PATH, index_col="state")
        extractor = FeatureExtractor(vs)
        predictor = CrimeRatePredictor()
        predictor.load(MODEL_PATH)
        run_demo(vs, extractor, feature_matrix, predictor)
