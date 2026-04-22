# 23k-0532
# Rayyan Ali

import os
import re
import math
from collections import defaultdict

import nltk
for _corpus in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
    nltk.download(_corpus, quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn


# use the same exact folder to run the code
# otherwise the code will not run

BASE         = os.path.dirname(os.path.abspath(__file__))
SPEECHES_DIR  = os.path.join(BASE, "Trump Speechs", "Trump Speechs")
STOPWORD_FILE = os.path.join(BASE, "Stopword-List.txt")
QUERY_FILE    = os.path.join(BASE, "Query List VSM.txt")
ALPHA = 0.005


def _lemmatize(word: str) -> str:
    return wn.morphy(word) or word


class Preprocessor:

    def __init__(self, stopword_file: str):
        self.stopwords = self._load_stopwords(stopword_file)

    def _load_stopwords(self, path: str) -> set:
        stopwords = set()
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip().lower()
                    if word:
                        stopwords.add(word)
        except FileNotFoundError:
            print(f"[Warning] Stop-word file not found: {path}")
        return stopwords

    def tokenize(self, text: str) -> list:
        return [t for t in word_tokenize(text) if t.isalpha()]

    def process(self, text: str) -> list:
        tokens = self.tokenize(text)
        result = []
        for tok in tokens:
            tok_lower = tok.lower()
            if tok_lower in self.stopwords:
                continue
            lemma = _lemmatize(tok_lower)
            if lemma and len(lemma) > 1:
                result.append(lemma)
        return result


def load_corpus(speeches_dir: str) -> dict:
    corpus = {}
    pattern = re.compile(r"speech_(\d+)\.txt$")
    for fname in sorted(os.listdir(speeches_dir)):
        m = pattern.match(fname)
        if m:
            doc_id = int(m.group(1))
            path = os.path.join(speeches_dir, fname)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                corpus[doc_id] = f.read()
    return corpus


class VSMIndex:

    def __init__(self):
        self.tf        = defaultdict(lambda: defaultdict(int))
        self.df        = defaultdict(int)
        self.idf       = {}
        self.tfidf     = defaultdict(dict)
        self.doc_norms = {}
        self.N         = 0
        self.vocabulary = []

    def build(self, corpus: dict, preprocessor: Preprocessor, min_tf: int = 1, min_df: int = 1):
        print(f"\n[Index] Building VSM over {len(corpus)} documents...")

        for doc_id, raw_text in corpus.items():
            tokens = preprocessor.process(raw_text)
            freq = defaultdict(int)
            for tok in tokens:
                freq[tok] += 1
            self.tf[doc_id] = dict(freq)

        self.N = len(corpus)

        for doc_id, freq_map in self.tf.items():
            for term in freq_map:
                self.df[term] += 1

        selected_terms = {
            term for term, df_val in self.df.items()
            if df_val >= min_df
            and any(self.tf[d].get(term, 0) >= min_tf for d in corpus)
        }

        for term in selected_terms:
            df_val = self.df[term]
            self.idf[term] = math.log(self.N / df_val) if df_val > 0 else 0.0

        for doc_id, freq_map in self.tf.items():
            vec = {}
            for term in selected_terms:
                raw_tf = freq_map.get(term, 0)
                if raw_tf >= min_tf:
                    vec[term] = raw_tf * self.idf.get(term, 0.0)
            self.tfidf[doc_id] = vec

        for doc_id, vec in self.tfidf.items():
            norm = math.sqrt(sum(v * v for v in vec.values()))
            self.doc_norms[doc_id] = norm if norm > 0 else 1.0

        self.vocabulary = sorted(selected_terms)
        print(f"[Index] Done. {len(self.vocabulary)} unique terms indexed.")


class QueryProcessor:

    def __init__(self, index: VSMIndex, preprocessor: Preprocessor):
        self.index        = index
        self.preprocessor = preprocessor

    def _query_vector(self, query_terms: list) -> dict:
        freq = defaultdict(int)
        for t in query_terms:
            freq[t] += 1
        q_vec = {}
        for term, tf_val in freq.items():
            if term in self.index.idf:
                q_vec[term] = tf_val * self.index.idf[term]
        return q_vec

    def _cosine_similarity(self, q_vec: dict, doc_id: int) -> float:
        doc_vec  = self.index.tfidf.get(doc_id, {})
        doc_norm = self.index.doc_norms.get(doc_id, 1.0)
        dot = sum(q_val * doc_vec.get(term, 0.0) for term, q_val in q_vec.items())
        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
        if q_norm == 0.0 or doc_norm == 0.0:
            return 0.0
        return dot / (q_norm * doc_norm)

    def search(self, query: str, alpha: float = ALPHA, top_k: int = None) -> list:
        query_terms = self.preprocessor.process(query)
        if not query_terms:
            return []
        q_vec = self._query_vector(query_terms)
        if not q_vec:
            return []
        scores = []
        for doc_id in self.index.tfidf:
            score = self._cosine_similarity(q_vec, doc_id)
            if score >= alpha:
                scores.append((doc_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None:
            scores = scores[:top_k]
        return scores


def parse_query_file(path: str) -> list:
    queries = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"[Warning] Query file not found: {path}")
        return queries

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r"(?:Query:|query\s*=\s*)['\"]?(.*?)['\"]?\s*$", line, re.IGNORECASE)
        if m:
            q_text = m.group(1).strip().strip("'\"")
            expected_len  = None
            expected_docs = set()
            j = i + 1
            while j < len(lines) and j < i + 6:
                lj = lines[j].strip()
                lm = re.search(r"Length\s*=\s*(\d+)", lj)
                if lm:
                    expected_len = int(lm.group(1))
                sm = re.search(r"\{([^}]+)\}", lj)
                if sm:
                    for tok in sm.group(1).split(","):
                        tok = tok.strip().strip("'\"")
                        if tok.isdigit():
                            expected_docs.add(int(tok))
                j += 1
            if q_text:
                queries.append({
                    "query":         q_text,
                    "expected_len":  expected_len,
                    "expected_docs": expected_docs,
                })
        i += 1
    return queries


def evaluate(retrieved: set, relevant: set) -> dict:
    if not retrieved:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp        = len(retrieved & relevant)
    precision = tp / len(retrieved) if retrieved else 0.0
    recall    = tp / len(relevant)  if relevant  else 0.0
    f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


BANNER = """
   Vector Space Model (VSM) – Information Retrieval       
     CS4051 | Spring 2026 | Trump Speeches (56 docs)       

"""

MENU = """
  [1]  Run all benchmark queries
  [2]  Free-text query search
  [3]  Inspect term TF-IDF weights
  [4]  Show index statistics
  [5]  Exit
"""


def run_cli(index: VSMIndex, preprocessor: Preprocessor, queries: list, alpha: float = ALPHA):
    ui_card = None

    qp = QueryProcessor(index, preprocessor)
    print(BANNER)
    print(f"  Loaded {index.N} documents | {len(index.vocabulary)} terms | alpha={alpha}")

    while True:
        print(MENU)
        choice = input("  Select option: ").strip()

        if choice == "1":
            _run_benchmark(qp, queries, alpha)
        elif choice == "2":
            _run_free_query(qp, alpha)
        elif choice == "3":
            _inspect_term(index, preprocessor)
        elif choice == "4":
            _show_stats(index)
        elif choice == "5":
            print("\n  Goodbye!\n")
            break
        else:
            print("  Invalid option – please enter 1 to 5.")


def _run_benchmark(qp: QueryProcessor, queries: list, alpha: float):
    print(f"\n  Running {len(queries)} benchmark queries (alpha={alpha})\n")
    sep = "  " + "─" * 82
    hdr = (f"  {'#':<4} {'Query':<42} {'Retr':>5} {'Gold':>5}"
           f" {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(hdr)
    print(sep)

    total_p = total_r = total_f = 0.0

    for idx, q_info in enumerate(queries, 1):
        q_text   = q_info["query"]
        expected = q_info["expected_docs"]
        exp_len  = q_info["expected_len"] or len(expected)

        results       = qp.search(q_text, alpha=alpha)
        retrieved_ids = {doc_id for doc_id, _ in results}
        metrics       = evaluate(retrieved_ids, expected) if expected else \
                        {"precision": 0, "recall": 0, "f1": 0}

        total_p += metrics["precision"]
        total_r += metrics["recall"]
        total_f += metrics["f1"]

        qt = q_text[:41]
        print(f"  {idx:<4} {qt:<42} {len(retrieved_ids):>5} {exp_len:>5}"
              f" {metrics['precision']:>6.3f} {metrics['recall']:>6.3f} {metrics['f1']:>6.3f}")

        for d, s in results:
            print(f"         speech_{d}({s:.3f})")

    n = len(queries)
    print(sep)
    print(f"  {'MACRO-AVERAGE':<48}"
          f" {total_p/n:>6.3f} {total_r/n:>6.3f} {total_f/n:>6.3f}\n")


def _run_free_query(qp: QueryProcessor, alpha: float):
    q_text = input("\n  Enter query: ").strip()
    if not q_text:
        return
    results = qp.search(q_text, alpha=alpha)
    if not results:
        print(f"\n  No documents found above alpha={alpha}.\n")
        return
    print(f"\n  Found {len(results)} document(s):\n")
    print(f"  {'Rank':<6} {'Document':<15} {'Cosine Score'}")
    print("  " + "─" * 35)
    for rank, (doc_id, score) in enumerate(results, 1):
        print(f"  {rank:<6} speech_{doc_id:<9} {score:.6f}")
    print()


def _inspect_term(index: VSMIndex, preprocessor: Preprocessor):
    term_raw = input("\n  Enter term to inspect: ").strip().lower()
    lemma    = _lemmatize(term_raw)
    print(f"\n  Input  : '{term_raw}'")
    print(f"  Lemma  : '{lemma}'")
    if lemma not in index.idf:
        print(f"  Term not in vocabulary.\n")
        return
    print(f"  IDF    : {index.idf[lemma]:.4f}")
    print(f"  DF     : {index.df[lemma]} / {index.N} docs")
    docs = [(d, v) for d, vec in index.tfidf.items()
            if (v := vec.get(lemma, 0)) > 0]
    docs.sort(key=lambda x: x[1], reverse=True)
    print(f"  Top documents by TF-IDF weight:")
    for doc_id, w in docs[:10]:
        tf_val = index.tf[doc_id].get(lemma, 0)
        print(f"    speech_{doc_id:<4}  TF={tf_val:3d}  TF-IDF={w:.4f}")
    print()


def _show_stats(index: VSMIndex):
    print(f"\n  Total documents : {index.N}")
    print(f"  Vocabulary size : {len(index.vocabulary)}")
    avg_terms = sum(len(v) for v in index.tfidf.values()) / index.N
    print(f"  Avg terms/doc   : {avg_terms:.1f}")
    top_idf = sorted(index.idf.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"  Top-10 by IDF   : {[t for t, _ in top_idf]}")
    top_df = sorted(index.df.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"  Top-10 by DF    : {[t for t, _ in top_df]}")
    print()


def main():
    print(BANNER)
    print("[Setup] Loading stop-words...")
    preprocessor = Preprocessor(STOPWORD_FILE)
    print(f"[Setup] {len(preprocessor.stopwords)} stop-words loaded.\n")

    print("[Corpus] Loading speech files...")
    corpus = load_corpus(SPEECHES_DIR)
    print(f"[Corpus] {len(corpus)} documents loaded.")

    index = VSMIndex()
    index.build(corpus, preprocessor, min_tf=1, min_df=1)

    queries = parse_query_file(QUERY_FILE)
    print(f"\n[Queries] {len(queries)} benchmark queries parsed.")

    run_cli(index, preprocessor, queries, alpha=ALPHA)


if __name__ == "__main__":
    main()
