import streamlit as st
import pyalex
from pyalex import Works
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import collections
import io
from docx import Document
import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
import re
import itertools
import math
import os
import time
from datetime import datetime, timezone


# ================= ANALYTICAL HELPERS =================

def _clean_text_spaces(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


def safe_betweenness(graph, k=200, weight=None):
    """Sampled betweenness; guarantees float for every node."""
    n = graph.number_of_nodes()
    if n == 0:
        return {}
    try:
        if n <= k:
            raw = nx.betweenness_centrality(graph, weight=weight, normalized=True)
        else:
            raw = nx.betweenness_centrality(
                graph, k=min(k, n), weight=weight, normalized=True, seed=42
            )
    except Exception:
        raw = {node: 0.0 for node in graph.nodes()}
    return {node: (float(v) if v is not None else 0.0) for node, v in raw.items()}


def calculate_network_silhouette(graph, communities, max_component_size=500):
    """Node-weighted silhouette from shortest-path distances on eligible components."""
    G = graph.to_undirected()
    node_to_cluster = {}
    for cluster_id, community in enumerate(communities):
        for node in community:
            node_to_cluster[node] = cluster_id

    node_scores = []
    cluster_score_values = collections.defaultdict(list)
    evaluated_nodes = 0
    eligible_components = 0
    skipped_components = 0

    for component in nx.connected_components(G):
        nodes = [n for n in component if n in node_to_cluster]
        if len(nodes) < 3:
            continue
        if len(nodes) > max_component_size:
            skipped_components += 1
            continue

        labels = np.array([node_to_cluster[n] for n in nodes])
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2 or len(unique_labels) >= len(nodes):
            continue

        subgraph = G.subgraph(nodes)
        paths = dict(nx.all_pairs_shortest_path_length(subgraph))
        n = len(nodes)
        distances = np.zeros((n, n), dtype=float)
        for i, source in enumerate(nodes):
            row = paths[source]
            for j, target in enumerate(nodes):
                distances[i, j] = row[target]

        masks = {lab: (labels == lab) for lab in unique_labels}
        for i, node in enumerate(nodes):
            own_label = labels[i]
            own_mask = masks[own_label].copy()
            own_mask[i] = False
            if not own_mask.any():
                score = 0.0
            else:
                a_i = float(distances[i, own_mask].mean())
                other_means = []
                for other_label in unique_labels:
                    if other_label == own_label:
                        continue
                    m = masks[other_label]
                    if m.any():
                        other_means.append(float(distances[i, m].mean()))
                if not other_means:
                    score = 0.0
                else:
                    b_i = min(other_means)
                    denom = max(a_i, b_i)
                    score = 0.0 if denom == 0 else (b_i - a_i) / denom
            node_scores.append(score)
            cluster_score_values[int(own_label)].append(score)

        evaluated_nodes += n
        eligible_components += 1

    global_score = float(np.mean(node_scores)) if node_scores else None
    cluster_scores = {
        cid: float(np.mean(vals)) for cid, vals in cluster_score_values.items() if vals
    }
    return {
        "global_silhouette": global_score,
        "evaluated_nodes": evaluated_nodes,
        "eligible_components": eligible_components,
        "skipped_components": skipped_components,
        "cluster_scores": cluster_scores,
    }


def kleinberg_burst_episodes(event_counts, exposures, years,
                             s=2.0, gamma=1.0, max_states=None):
    """Kleinberg-style hierarchical Poisson burst detection."""
    x = np.asarray(event_counts, dtype=float)
    n = np.asarray(exposures, dtype=float)
    years = list(years)

    if len(x) == 0 or len(x) != len(n) or len(x) != len(years):
        return []
    if np.sum(x) <= 0 or np.sum(n) <= 0:
        return []

    n = np.maximum(n, 1.0)
    base_rate = float(np.sum(x) / np.sum(n))
    if base_rate <= 0:
        return []

    if max_states is None:
        max_states = 3 if len(years) < 15 else 4

    states = np.arange(max_states + 1)
    transition_penalty = gamma * math.log(max(2, len(years)))

    costs = np.zeros((len(years), len(states)), dtype=float)
    for t in range(len(years)):
        for k, state in enumerate(states):
            mu = max(base_rate * n[t] * (s ** int(state)), 1e-12)
            costs[t, k] = mu - x[t] * math.log(mu)

    dp = np.full_like(costs, np.inf)
    back = np.zeros_like(costs, dtype=int)
    dp[0, :] = costs[0, :]

    for t in range(1, len(years)):
        prev = dp[t - 1, :]
        for k, state in enumerate(states):
            transition_costs = prev + transition_penalty * np.abs(states - state)
            best = int(np.argmin(transition_costs))
            dp[t, k] = costs[t, k] + transition_costs[best]
            back[t, k] = best

    path = np.zeros(len(years), dtype=int)
    path[-1] = int(np.argmin(dp[-1, :]))
    for t in range(len(years) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]

    episodes = []
    start = None
    for i, state in enumerate(path):
        if state > 0 and start is None:
            start = i
        is_end = start is not None and (state == 0 or i == len(path) - 1)
        if is_end:
            end = i - 1 if state == 0 else i
            episode_states = path[start:end + 1]
            episode_events = x[start:end + 1]
            episodes.append({
                "Begin": int(years[start]),
                "End": int(years[end]),
                "Burst Level": int(np.max(episode_states)),
                "Events During Burst": int(np.sum(episode_events)),
                "Duration (years)": int(end - start + 1),
            })
            start = None

    return episodes


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_batch_cached(ids_tuple, max_retries=4):
    ids = list(ids_tuple)
    short_ids = [i.split("/")[-1] for i in ids]
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = Works().filter(openalex="|".join(short_ids)).get(per_page=len(short_ids))
            return list(resp)
        except Exception as e:
            last_exc = e
            time.sleep(0.5 * (2 ** attempt))
    raise last_exc


def fetch_reference_metadata(reference_ids_tuple, batch_size=20):
    selected_ids = list(reference_ids_tuple)
    metadata = {}
    errors = []
    total_batches = (len(selected_ids) + batch_size - 1) // batch_size
    progress = st.progress(0.0, text="Fetching cited-work metadata…")
    for idx, offset in enumerate(range(0, len(selected_ids), batch_size)):
        batch = selected_ids[offset:offset + batch_size]
        if not batch:
            continue
        try:
            for item in _fetch_batch_cached(tuple(batch)):
                item_id = item.get("id")
                if item_id:
                    metadata[item_id] = item
        except Exception as e:
            errors.append(f"batch {idx} ({len(batch)} IDs): {type(e).__name__}: {str(e)[:120]}")
        progress.progress(
            min(1.0, (idx + 1) / max(1, total_batches)),
            text=f"Fetched {len(metadata)}/{len(selected_ids)} cited works…",
        )
        time.sleep(0.15)
    progress.empty()
    return metadata, len(selected_ids), errors


def build_author_cocitation_analysis(corpus_works, reference_metadata):
    """Build genuine ACA graph from co-occurring cited authors in corpus reference lists."""
    ACA = nx.Graph()
    author_names = {}
    author_citing_documents = collections.Counter()

    for citing_work in corpus_works:
        cited_author_ids = set()
        for ref_id in citing_work.get("referenced_works", []) or []:
            cited_work = reference_metadata.get(ref_id)
            if not cited_work:
                continue
            for authorship in cited_work.get("authorships", []) or []:
                author = authorship.get("author") or {}
                author_id = author.get("id")
                author_name = author.get("display_name")
                if author_id and author_name:
                    cited_author_ids.add(author_id)
                    author_names[author_id] = _clean_text_spaces(author_name)

        for author_id in cited_author_ids:
            author_citing_documents[author_id] += 1
            ACA.add_node(author_id, name=author_names.get(author_id, author_id))

        for a_id, b_id in itertools.combinations(sorted(cited_author_ids), 2):
            if ACA.has_edge(a_id, b_id):
                ACA[a_id][b_id]["weight"] += 1
            else:
                ACA.add_edge(a_id, b_id, weight=1)

    if ACA.number_of_nodes() > 0:
        distance_graph = ACA.copy()
        for u, v, data in distance_graph.edges(data=True):
            data["distance"] = 1.0 / max(1, data.get("weight", 1))
        betweenness = safe_betweenness(distance_graph, weight="distance")
    else:
        betweenness = {}

    rows = []
    ranked = sorted(
        ACA.nodes(),
        key=lambda node: (
            author_citing_documents[node],
            ACA.degree(node, weight="weight"),
        ),
        reverse=True,
    )
    for rank, author_id in enumerate(ranked[:15], 1):
        link_strength = int(ACA.degree(author_id, weight="weight"))
        citing_docs = int(author_citing_documents[author_id])
        link_per_doc = round(float(link_strength) / max(1, int(citing_docs)), 2)
        btw = round(float(betweenness.get(author_id, 0.0)), 5)

        expected_lpd = round(link_strength / max(1, citing_docs), 2)
        if abs(float(link_per_doc) - float(expected_lpd)) > 1e-6:
            raise RuntimeError(
                f"ACA integrity check failed for {author_names.get(author_id, author_id)}: "
                f"Link/Doc={link_per_doc} != {link_strength}/{citing_docs}={expected_lpd}"
            )

        rows.append({
            "Rank": rank,
            "Co-Cited Author": author_names.get(author_id, author_id),
            "Citing Corpus Documents": citing_docs,
            "Co-Citation Link Strength": link_strength,
            "Link / Citing Doc": float(link_per_doc),
            "Betweenness Centrality": btw,
            "OpenAlex Author ID": author_id,
        })

    return ACA, rows, author_citing_documents


def make_year_series(counter, years):
    return [int(counter.get(year, 0)) for year in years]


STOP_WORDS = {
    "and", "the", "of", "in", "a", "to", "for", "is", "on", "with", "by",
    "as", "an", "at", "from", "it", "or", "using", "study", "analysis",
    "review", "based", "research", "paper", "article", "case", "results",
    "findings", "data", "approach", "role", "effect", "effects", "impact",
    "impacts", "new", "toward", "towards", "via", "across", "within",
    "between", "among", "under", "over", "into", "about", "their", "its",
    "these", "those", "such", "been", "have", "has", "was", "were", "are",
    "can", "may", "might", "will", "would", "should", "could", "more",
    "most", "some", "any", "all", "how", "what", "when", "where", "why",
    "which", "who", "this", "that", "other", "also", "than", "then",
    "only", "very", "much", "many", "well", "both", "each", "either",
    "however", "therefore", "thus", "example", "table", "figure",
    "chapter", "section", "method", "methods", "methodology", "used", "uses",
}

BOOLEAN_WORDS = {"and", "or", "not"}

NON_SUBSTANTIVE_TITLES = {
    "front matter", "preface", "foreword", "editorial", "briefings",
    "erratum", "corrigendum", "retraction", "table of contents",
    "cover page", "title page", "acknowledgements", "acknowledgments",
    "news from here and there", "index", "author index", "subject index",
}

CONCEPT_PAREN_PATTERN = re.compile(r"\s*\([^)]*\)\s*")


def clean_title(raw_title):
    """Cleans whitespace and collapses exact back-to-back repeated titles."""
    t = _clean_text_spaces(raw_title)
    if not t:
        return "Untitled"
    half = len(t) // 2
    if len(t) % 2 == 0 and t[:half].strip().lower() == t[half:].strip().lower():
        return t[:half].strip()
    match = re.match(r"^(.{8,}?)\s+\1$", t, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return t


def is_substantive_work(work):
    """Filters out empty or generic front-matter / paratext entries across any faculty."""
    cleaned = clean_title(work.get("title")).lower()
    stripped = re.sub(r"[^\w\s]", "", cleaned).strip()
    if not stripped or stripped in NON_SUBSTANTIVE_TITLES:
        return False
    return True


def normalize_term(raw_term):
    if not raw_term:
        return None
    cleaned = CONCEPT_PAREN_PATTERN.sub(" ", str(raw_term)).strip()
    cleaned = _clean_text_spaces(cleaned)
    low = cleaned.lower()
    if not low or low in STOP_WORDS or len(low) < 2:
        return None

    words = low.split()
    norm_words = []
    for w in words:
        if len(w) > 4 and w.endswith("ies") and not w.endswith(("series", "species")):
            w = w[:-3] + "y"
        elif (
            len(w) > 4
            and w.endswith("s")
            and not w.endswith(("ss", "us", "is", "as", "os", "ics"))
        ):
            w = w[:-1]
        norm_words.append(w)

    canonical_low = " ".join(norm_words)
    if canonical_low in STOP_WORDS:
        return None

    formatted = " ".join(
        tok.upper() if (tok.isupper() and len(tok) <= 6) else tok.capitalize()
        for tok in canonical_low.split()
    )
    return formatted


def extract_document_terms(work, max_terms=12):
    doc_term_counter = collections.Counter()
    for c in work.get("concepts") or []:
        if c.get("score", 0) >= 0.5:
            norm = normalize_term(c.get("display_name"))
            if norm:
                doc_term_counter[norm] += 2
    title_text = clean_title(work.get("title")).lower()
    for w in re.findall(r"\b[a-zA-Z]{3,}\b", title_text):
        if w not in STOP_WORDS:
            norm = normalize_term(w)
            if norm:
                doc_term_counter[norm] += 1
    return {t for t, _ in doc_term_counter.most_common(max_terms)}


def classify_thematic_keyword(kw):
    """Faculty-neutral thematic grouping based on general academic/methodological roles."""
    k = kw.lower().strip()

    def has_any(patterns):
        return any(re.search(p, k) for p in patterns)

    if has_any([
        r"\brank(ing|s)?\b", r"\bcitation(s)?\b", r"\bindex\b", r"\bindicator(s)?\b",
        r"\bmetric(s)?\b", r"\bimpact\b", r"\bscore(s)?\b", r"\bprestige\b",
        r"\breputation\b", r"\bvisibility\b", r"\bproductivity\b",
    ]):
        return "Indicators, Metrics & Ranking Terms"

    if has_any([
        r"\bmodel(s|ing)?\b", r"\bmethod(s|ology)?\b", r"\btechnique(s)?\b",
        r"\balgorithm(s)?\b", r"\bframework(s)?\b", r"\bnetwork(s)?\b",
        r"\bgraph(s)?\b", r"\bregression\b", r"\bbayesian\b", r"\bfuzzy\b",
        r"\bsimulation\b", r"\bstatistic(s|al)?\b", r"\banalysis\b", r"\bmeasure(ment)?\b",
    ]):
        return "Methods, Models & Analytical Terms"

    if has_any([
        r"\bevaluat(e|ion|ing)?\b", r"\bassess(ment|ing)?\b", r"\bqualit(y|ies)\b",
        r"\bperformance\b", r"\bbenchmark(s|ing)?\b", r"\baccreditation\b",
        r"\befficiency\b", r"\bstandard(s|ization)?\b", r"\bvalidity\b",
    ]):
        return "Evaluation, Quality & Performance"

    if has_any([
        r"\binstitution(s|al)?\b", r"\buniversit(y|ies)\b", r"\beducation(al)?\b",
        r"\bsystem(s)?\b", r"\bpolicy\b", r"\bgovernance\b", r"\bcontext\b",
        r"\bregional\b", r"\bnational\b", r"\bglobal\b", r"\bcommunity\b",
    ]):
        return "Systems, Institutions & Contexts"

    return "Domain & Topical Keywords"


def _reconstruct_abstract(work):
    inv = work.get("abstract_inverted_index")
    if not isinstance(inv, dict) or not inv:
        return ""
    positions = []
    for word, pos_list in inv.items():
        if isinstance(pos_list, list):
            for p in pos_list:
                positions.append((p, word))
    positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in positions)


def _normalize_search_text(text):
    t = re.sub(r"[-‐‑‒–—]+", " ", (text or "").lower())
    return _clean_text_spaces(t)


def _parse_query_entries(raw_lines):
    """
    Parses each keyword line into:
      - exact phrase match when enclosed in "..." or '...'
      - word-boundary tokens for unquoted terms
    """
    entries = []
    for line in raw_lines:
        s = line.strip()
        if not s:
            continue
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', s)
        phrases = [
            _normalize_search_text(q1 or q2)
            for q1, q2 in quoted
            if _normalize_search_text(q1 or q2)
        ]
        remainder = re.sub(r'"[^"]+"|\'[^\']+\'', " ", s)
        tokens = [
            _normalize_search_text(tok)
            for tok in re.findall(r"\b[a-zA-Z0-9]+\b", remainder)
            if tok.lower() not in STOP_WORDS and tok.lower() not in BOOLEAN_WORDS and len(tok) >= 2
        ]
        if phrases or tokens:
            entries.append({"phrases": phrases, "tokens": tokens})
    return entries


def build_openalex_search_query(raw_lines, match_mode="any", exact_phrase=False, token_logic="all"):
    """Builds a clean OpenAlex search string from the user's exact keywords."""
    cleaned_items = []
    for line in raw_lines:
        item = line.strip()
        if not item:
            continue
        is_quoted = (item.startswith('"') and item.endswith('"')) or (
            item.startswith("'") and item.endswith("'")
        )
        if exact_phrase and not is_quoted and " " in item:
            item = f'"{item}"'
        elif not exact_phrase and not is_quoted and " " in item and token_logic == "any":
            toks = [
                t for t in re.findall(r"\b[a-zA-Z0-9]+\b", item)
                if t.lower() not in STOP_WORDS and t.lower() not in BOOLEAN_WORDS
            ]
            if len(toks) > 1:
                item = "(" + " OR ".join(toks) + ")"
        cleaned_items.append(item)
    joiner = " AND " if match_mode == "all" else " OR "
    return joiner.join(cleaned_items)


def make_relevance_filter(
    raw_lines, match_mode, search_scope="title_only", exact_phrase=False, token_logic="all"
):
    """
    Keyword filter aligned with the user's actual keywords and logic settings.
    Does not apply arbitrary 6-char stemming or hardcoded faculty/domain rules.
    """
    entries = _parse_query_entries(raw_lines)
    if exact_phrase:
        for ent in entries:
            if not ent["phrases"] and ent["tokens"]:
                ent["phrases"] = [" ".join(ent["tokens"])]
                ent["tokens"] = []
    if not entries:
        return None

    def _entry_matches(entry, norm_text):
        if not norm_text:
            return False
        for phrase in entry["phrases"]:
            pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
            if not re.search(pattern, norm_text):
                return False
        if entry["tokens"]:
            tok_matches = [
                bool(re.search(r"\b" + re.escape(tok) + r"\b", norm_text))
                for tok in entry["tokens"]
            ]
            if token_logic == "any":
                if not any(tok_matches):
                    return False
            else:
                if not all(tok_matches):
                    return False
        return True

    def _check(work):
        title = _normalize_search_text(clean_title(work.get("title") or ""))
        if search_scope == "title_only":
            target = title
        else:
            abstract = _normalize_search_text(_reconstruct_abstract(work))
            concepts = _normalize_search_text(" ".join(
                (c.get("display_name") or "")
                for c in (work.get("concepts") or [])
                if c.get("score", 0) >= 0.5
            ))
            target = f"{title} {abstract} {concepts}".strip()

        hits = [_entry_matches(ent, target) for ent in entries]
        return all(hits) if match_mode == "all" else any(hits)

    return _check


def _title_signature(title):
    toks = [
        t for t in re.findall(r"\w+", clean_title(title).lower())
        if t not in STOP_WORDS and len(t) > 2
    ]
    return frozenset(toks)


def deduplicate_works(works_list):
    by_id = {}
    no_id = []
    for w in works_list:
        wid = w.get("id")
        if wid:
            prev = by_id.get(wid)
            if prev is None or w.get("cited_by_count", 0) > prev.get("cited_by_count", 0):
                by_id[wid] = w
        else:
            no_id.append(w)

    combined = list(by_id.values()) + no_id
    combined.sort(key=lambda w: w.get("cited_by_count", 0), reverse=True)

    seen_sigs = []
    unique_works = []
    for w in combined:
        sig = _title_signature(w.get("title"))
        is_dup = False
        if sig and len(sig) >= 4:
            for s in seen_sigs:
                if len(sig & s) / max(1, len(sig | s)) >= 0.85:
                    is_dup = True
                    break
        if is_dup:
            continue
        if sig and len(sig) >= 4:
            seen_sigs.append(sig)
        unique_works.append(w)
    return unique_works


def get_cluster_label_terms(member_works, corpus_term_freq, corpus_size, top_n=3):
    cluster_term_freq = collections.Counter()
    for mw in member_works:
        for term in extract_document_terms(mw, max_terms=12):
            cluster_term_freq[term] += 1

    scored = []
    for term, freq in cluster_term_freq.items():
        tf = freq / max(1, len(member_works))
        idf = math.log((1 + corpus_size) / (1 + corpus_term_freq.get(term, 1))) + 1.0
        scored.append((term, tf * idf))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scored[:top_n]]


# ================= STREAMLIT CONFIG =================
st.set_page_config(
    page_title="NAGraph — Bibliometric & Citation Network Analysis Suite",
    layout="wide",
)

st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #1E293B; color: #F8FAFC; font-size: 18px;
        font-weight: bold; padding: 10px 24px; border-radius: 6px;
        border: 1px solid #475569; width: 100%;
    }
    div.stButton > button:first-child:hover {
        background-color: #334155; border-color: #64748B; color: #FFFFFF;
    }
    [data-testid="stSidebar"] div.block-container {
        padding-top: 1rem; padding-bottom: 1rem;
    }
    [data-testid="stSidebar"] hr { margin-top: 0.8rem; margin-bottom: 0.8rem; }
    [data-testid="stSidebar"] img { margin-bottom: -10px; }
    </style>
""",
    unsafe_allow_html=True,
)

# ================= SIDEBAR =================
if os.path.exists("nagraph_logo.png"):
    st.sidebar.image("nagraph_logo.png", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Developed by:<br>**Dr. Nibu A. George,**<br>Associate Professor of Physics,<br>"
    "(IQAC Coordinator),<br>Baselius College Kottayam, (Autonomous).",
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
st.sidebar.subheader("📖 User Guide & Manual")
try:
    with open("nagraph user manual.docx", "rb") as docx_file:
        st.sidebar.download_button(
            label="📥 Download Help Manual (.docx)",
            data=docx_file,
            file_name="nagraph user manual.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
except FileNotFoundError:
    st.sidebar.warning("`nagraph user manual.docx` not found in directory.")

st.sidebar.markdown("---")

st.title("NAGraph: Bibliometric & Citation Network Analysis Suite")
st.markdown(
    "NAGraph is a faculty-neutral bibliometric and citation-network analysis suite. "
    "It retrieves OpenAlex records for any discipline and analyses citation networks, "
    "communities, keyword co-occurrence, temporal bursts, author co-citation, and publication metadata."
)

with st.expander("📖 Guide & Glossary: How to Read This Analysis"):
    st.markdown("""
    ### 🔍 Parameters
    * **Cluster Analysis:** Modularity Q, shortest-path silhouette on eligible components, and empirical mean publication years.
    * **Keyword Centrality & Bursts:** Document term frequency, co-occurrence betweenness centrality, and Kleinberg-style temporal bursts.
    * **Author Co-Citation Analysis (ACA):** Co-cited-author links derived from retrieved cited-work metadata.
    * **Institutional / Country / Source Tables:** Publication output, mean year, and international co-authorship share.
    * **Exploratory Composite Score:** 35% PageRank + 25% In-Network cites + 25% Global log-citations + 15% Betweenness (with weight-sensitivity check).
    """)

# ================= SIDEBAR SEARCH SETTINGS =================
st.sidebar.header("Search settings")

keywords_input = st.sidebar.text_area(
    "Topic keywords (one per line or comma-separated; use \"quotes\" for exact phrases)",
    value="", height=100,
)
match_mode = st.sidebar.radio("Across lines/keywords", options=["any", "all"], horizontal=True)
token_logic = st.sidebar.radio(
    "Within multi-word line",
    options=["all", "any"],
    horizontal=True,
    help="Use 'all' for stricter multi-word matching or 'any' to broaden recall across words in a line.",
)
exact_phrase_mode = st.sidebar.checkbox(
    "Treat multi-word keywords as exact phrases", value=True
)
search_scope = st.sidebar.selectbox(
    "Keyword relevance scope",
    options=["title_only", "title_abstract_concepts"],
    index=1,
    format_func=lambda s: "Title only (strictest)" if s == "title_only" else "Title, abstract & concepts",
)
filter_paratext = st.sidebar.checkbox(
    "Exclude non-article front matter / paratext titles", value=True
)
pub_years = st.sidebar.slider(
    "Publication years", min_value=1950, max_value=2026, value=(2008, 2025)
)
max_results = st.sidebar.slider(
    "Works to retrieve", min_value=10, max_value=500, value=400
)
node_cap = st.sidebar.slider(
    "Node cap (max works in network)", min_value=50, max_value=500, value=450
)
sort_option = st.sidebar.selectbox(
    "Retrieve by", options=["relevance", "cited_by_count", "publication_date"]
)
add_cited_refs = st.sidebar.checkbox("Add highly cited references (one hop)", value=True)
boost_latest = st.sidebar.checkbox(
    "Recent-work boost (extra search of the last 3 years)", value=False
)
time_slice_years = st.sidebar.slider(
    "Timeline slice window (years per slice)", min_value=1, max_value=10, value=3
)

DEFAULT_OPENALEX_API_KEY = "XpKzLDClirFYfMStLb8T4e"
try:
    configured_openalex_key = st.secrets.get("OPENALEX_API_KEY", "")
except Exception:
    configured_openalex_key = ""
configured_openalex_key = (
    configured_openalex_key
    or os.environ.get("OPENALEX_API_KEY", "")
    or DEFAULT_OPENALEX_API_KEY
)

api_key_input = st.sidebar.text_input(
    "OpenAlex API key (free, recommended)",
    value=configured_openalex_key, type="password",
    help="Pre-filled by default; override here or via .streamlit/secrets.toml / OPENALEX_API_KEY env var.",
)
user_email = st.sidebar.text_input("Your Email (for OpenAlex Polite Pool)")

reference_metadata_cap = st.sidebar.slider(
    "Maximum cited-reference records for ACA/burst analysis",
    min_value=100, max_value=2000, value=600, step=100,
)

# ==========================================================
# EXECUTION
# ==========================================================
if st.sidebar.button("Run analysis"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    if not user_email or "example.com" in user_email:
        st.warning("Please provide a valid email address for OpenAlex rate-limiting compliance.")
    elif not keywords_input.strip():
        st.warning("Please enter at least one keyword before running the analysis.")
    else:
        pyalex.config.email = user_email
        if api_key_input.strip():
            pyalex.config.api_key = api_key_input.strip()
        else:
            st.info("No OpenAlex API key detected. Configure OPENALEX_API_KEY for authenticated access.")

        start_yr = max(1950, min(2026, int(pub_years[0])))
        end_yr = max(start_yr, min(2026, int(pub_years[1])))

        retrieval_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        raw_lines = [line.strip() for line in re.split(r"[\n,]+", keywords_input) if line.strip()]
        cleaned_keywords = build_openalex_search_query(
            raw_lines,
            match_mode=match_mode,
            exact_phrase=exact_phrase_mode,
            token_logic=token_logic,
        )
        display_keywords = "; ".join(raw_lines)

        is_relevant = make_relevance_filter(
            raw_lines,
            match_mode=match_mode,
            search_scope=search_scope,
            exact_phrase=exact_phrase_mode,
            token_logic=token_logic,
        )

        def _accept_work(w):
            if filter_paratext and not is_substantive_work(w):
                return False
            if is_relevant is not None and not is_relevant(w):
                return False
            return True

        filter_description = (
            f"scope={search_scope}, mode={match_mode}, token_logic={token_logic}, "
            f"exact_phrase={exact_phrase_mode}, paratext_excluded={filter_paratext}"
        )

        with st.status("Running NAGraph analysis…", expanded=True) as status:
            try:
                # ---- 1. Main retrieval ----
                status.write("📥 Retrieving works from OpenAlex…")
                results = []
                target_count = max_results
                batch_size = 50
                query = Works().search(cleaned_keywords).filter(
                    publication_year=f"{start_yr}-{end_yr}"
                )
                if sort_option == "cited_by_count":
                    query = query.sort(cited_by_count="desc")
                elif sort_option == "publication_date":
                    query = query.sort(publication_date="desc")

                openalex_total_matches = None
                try:
                    meta_probe = query.get(per_page=1)
                    meta = getattr(meta_probe, "meta", None)
                    if meta is not None:
                        openalex_total_matches = meta.count
                except Exception:
                    openalex_total_matches = None

                pager = query.paginate(per_page=batch_size)
                for page in pager:
                    for p in page:
                        results.append(p)
                        if len(results) >= target_count:
                            break
                    if len(results) >= target_count:
                        break
                results = results[:target_count]

                results = [w for w in results if _accept_work(w)]
                results = deduplicate_works(results)
                base_retrieved_count = len(results)
                status.write(f"   → Base retrieval (keyword-matched & deduplicated): {base_retrieved_count} works")

                # ---- 2. Recent boost ----
                recent_added_count = 0
                if boost_latest:
                    recent_start = max(start_yr, end_yr - 2)
                    try:
                        recent_query = Works().search(cleaned_keywords).filter(
                            publication_year=f"{recent_start}-{end_yr}"
                        )
                        existing_ids = {w.get("id") for w in results}
                        existing_sigs = [_title_signature(w.get("title")) for w in results]
                        for page in recent_query.paginate(per_page=50):
                            for rw in page:
                                r_id = rw.get("id")
                                r_sig = _title_signature(rw.get("title"))
                                is_dup = False
                                if r_sig and len(r_sig) >= 4:
                                    for s in existing_sigs:
                                        if len(r_sig & s) / max(1, len(r_sig | s)) >= 0.85:
                                            is_dup = True
                                            break
                                if (
                                    r_id not in existing_ids
                                    and not is_dup
                                    and _accept_work(rw)
                                ):
                                    results.append(rw)
                                    existing_ids.add(r_id)
                                    if r_sig and len(r_sig) >= 4:
                                        existing_sigs.append(r_sig)
                                    recent_added_count += 1
                            if recent_added_count >= 100:
                                break
                    except Exception as e:
                        st.warning(f"Recent-work boost failed: {e}")
                status.write(f"   → Recent-work boost added: {recent_added_count}")

                # ---- 3. One-hop references (filtered by keyword relevance & substantive check) ----
                added_ref_count = 0
                if add_cited_refs:
                    ref_counter = collections.Counter()
                    work_id_to_work = {w.get("id"): w for w in results}
                    for work in results:
                        for ref_id in work.get("referenced_works", []):
                            if ref_id not in work_id_to_work:
                                ref_counter[ref_id] += 1
                    frequent_refs = [r for r, c in ref_counter.items() if c >= 2]
                    if frequent_refs:
                        try:
                            fetched_refs = Works().filter(
                                openalex="|".join(frequent_refs[:40])
                            ).get(per_page=40)
                            existing_ids = {w.get("id") for w in results}
                            existing_sigs = [_title_signature(w.get("title")) for w in results]
                            for fr in fetched_refs:
                                fid = fr.get("id")
                                f_sig = _title_signature(fr.get("title"))
                                is_dup = False
                                if f_sig and len(f_sig) >= 4:
                                    for s in existing_sigs:
                                        if len(f_sig & s) / max(1, len(f_sig | s)) >= 0.85:
                                            is_dup = True
                                            break
                                py = fr.get("publication_year")
                                if (
                                    fid
                                    and fid not in existing_ids
                                    and not is_dup
                                    and py
                                    and start_yr <= int(py) <= end_yr
                                    and _accept_work(fr)
                                ):
                                    results.append(fr)
                                    existing_ids.add(fid)
                                    if f_sig and len(f_sig) >= 4:
                                        existing_sigs.append(f_sig)
                                    added_ref_count += 1
                        except Exception as e:
                            st.warning(f"One-hop reference expansion failed: {e}")
                status.write(f"   → One-hop references added (matching keywords): {added_ref_count}")

                total_candidates = len(results)
                dropped_works_count = max(0, total_candidates - node_cap)

                if len(results) > node_cap:
                    pool_ids = {w.get("id") for w in results}
                    in_cites_map = collections.defaultdict(int)
                    for w in results:
                        for ref in w.get("referenced_works", []):
                            if ref in pool_ids:
                                in_cites_map[ref] += 1

                    def priority_key(w):
                        in_net = in_cites_map.get(w.get("id"), 0)
                        global_c = np.log1p(w.get("cited_by_count", 0))
                        return 0.6 * in_net + 0.4 * global_c

                    results = sorted(results, key=priority_key, reverse=True)[:node_cap]

                if not results:
                    status.update(label="No matching results", state="error")
                    st.info(
                        "No publications matched your keyword criteria in the selected year range. "
                        "Try broadening search scope or match mode."
                    )
                    st.stop()

                st.success(
                    f"Compiled {len(results)} publications "
                    f"(Base: {base_retrieved_count}, Recent: {recent_added_count}, One-Hop: {added_ref_count})."
                )
                st.session_state["search_results"] = results
                st.session_state["search_query"] = display_keywords
                st.session_state["openalex_query_string"] = cleaned_keywords
                st.session_state["retrieval_timestamp"] = retrieval_timestamp
                st.session_state["run_params"] = {
                    "match_mode": match_mode,
                    "token_logic": token_logic,
                    "exact_phrase_mode": exact_phrase_mode,
                    "search_scope": search_scope,
                    "filter_paratext": filter_paratext,
                    "start_yr": start_yr,
                    "end_yr": end_yr,
                    "max_results": max_results,
                    "node_cap": node_cap,
                    "reference_metadata_cap": reference_metadata_cap,
                    "sort_option": sort_option,
                    "add_cited_refs": add_cited_refs,
                    "boost_latest": boost_latest,
                    "time_slice_years": time_slice_years,
                }
                st.session_state["coverage_stats"] = {
                    "base": base_retrieved_count,
                    "recent": recent_added_count,
                    "one_hop": added_ref_count,
                    "total_candidates": total_candidates,
                    "dropped": dropped_works_count,
                    "openalex_total_matches": openalex_total_matches,
                    "filter_active": is_relevant is not None,
                    "filter_description": filter_description,
                }

                # ---- Citation graph ----
                status.write("🕸️ Building citation network…")
                G = nx.DiGraph()
                work_ids = {w.get("id") for w in results}
                for work in results:
                    w_id = work.get("id")
                    if w_id:
                        G.add_node(w_id, title=clean_title(work.get("title")),
                                   cites=work.get("cited_by_count", 0))
                        for ref in work.get("referenced_works", []):
                            if ref in work_ids:
                                G.add_edge(w_id, ref)

                pagerank = nx.pagerank(G, alpha=0.85) if G.number_of_nodes() > 0 else {}
                betweenness = safe_betweenness(G)
                in_cites_counts = {n: G.in_degree(n) for n in G.nodes()}
                out_refs_counts = {n: G.out_degree(n) for n in G.nodes()}

                pr_vals = list(pagerank.values()) or [1]
                bw_vals = list(betweenness.values()) or [1]
                in_vals = list(in_cites_counts.values()) or [1]
                glob_vals = [np.log1p(w.get("cited_by_count", 0)) for w in results] or [1]
                max_pr = max(pr_vals) or 1
                max_bw = max(bw_vals) or 1
                max_in = max(in_vals) or 1
                max_glob = max(glob_vals) or 1

                in_list = list(in_cites_counts.values()) or [0]
                out_list = list(out_refs_counts.values()) or [0]
                positive_bw_list = [v for v in betweenness.values() if v > 0]
                p75_in = np.percentile(in_list, 75) if in_list else 0
                p90_out = np.percentile(out_list, 90) if out_list else 0
                p90_bw = np.percentile(positive_bw_list, 75) if positive_bw_list else 0
                mid_year = start_yr + (end_yr - start_yr) / 2

                scored_works = []
                for work in results:
                    w_id = work.get("id")
                    pr = pagerank.get(w_id, 0) / max_pr
                    bw = betweenness.get(w_id, 0) / max_bw
                    inc = in_cites_counts.get(w_id, 0) / max_in
                    glob = np.log1p(work.get("cited_by_count", 0)) / max_glob
                    score = round(100 * (0.35 * pr + 0.25 * inc + 0.25 * glob + 0.15 * bw), 2)

                    yr = work.get("publication_year", 2000)
                    inc_val = in_cites_counts.get(w_id, 0)
                    out_val = out_refs_counts.get(w_id, 0)
                    bw_val = betweenness.get(w_id, 0)
                    roles = []
                    if yr <= mid_year and inc_val > 0 and inc_val >= p75_in:
                        roles.append("Foundational")
                    if out_val >= p90_out and out_val >= 10:
                        roles.append("Review / hub")
                    if bw_val > 0 and bw_val >= p90_bw:
                        roles.append("Bridge")
                    if yr >= (end_yr - 2):
                        roles.append("Recent influential")
                    if not roles:
                        roles.append("Supporting")
                    scored_works.append((score, work, ", ".join(roles)))

                scored_works.sort(key=lambda x: x[0], reverse=True)
                st.session_state["scored_works"] = scored_works

                # ---- Score sensitivity ----
                def _score_with(weights, wid):
                    pr = pagerank.get(wid, 0) / max_pr
                    bw = betweenness.get(wid, 0) / max_bw
                    inc = in_cites_counts.get(wid, 0) / max_in
                    glob = np.log1p(
                        next((w.get("cited_by_count", 0) for w in results
                              if w.get("id") == wid), 0)
                    ) / max_glob
                    return (
                        weights["pr"] * pr + weights["inc"] * inc
                        + weights["glob"] * glob + weights["bw"] * bw
                    )

                weight_sets = {
                    "baseline (35/25/25/15)": {"pr": 0.35, "inc": 0.25, "glob": 0.25, "bw": 0.15},
                    "citations-only":         {"pr": 0.00, "inc": 0.00, "glob": 1.00, "bw": 0.00},
                    "equal-weight":           {"pr": 0.25, "inc": 0.25, "glob": 0.25, "bw": 0.25},
                    "network-only":           {"pr": 0.40, "inc": 0.40, "glob": 0.00, "bw": 0.20},
                }
                top15_baseline = {w.get("id") for _, w, _ in scored_works[:15]}
                rank_agreement = {}
                for name, wts in weight_sets.items():
                    ranked = sorted(
                        results,
                        key=lambda w: _score_with(wts, w.get("id")),
                        reverse=True,
                    )[:15]
                    ids = {w.get("id") for w in ranked}
                    overlap = len(ids & top15_baseline)
                    rank_agreement[name] = {
                        "top15_overlap_with_baseline": overlap,
                        "overlap_percent": round(100 * overlap / 15, 1),
                    }
                st.session_state["score_sensitivity"] = rank_agreement

                # ---- Entity stats ----
                status.write("📊 Parsing entity statistics…")
                inst_stats = collections.defaultdict(lambda: {"count": 0, "years": []})
                country_stats = collections.defaultdict(
                    lambda: {"count": 0, "years": [], "intl": 0}
                )
                country_coauth_edges = collections.Counter()
                journal_stats = collections.defaultdict(lambda: {"count": 0})
                author_stats = collections.defaultdict(
                    lambda: {"name": "", "works": 0, "cites": 0, "years": [], "work_entries": []}
                )
                kw_freq = collections.Counter()
                keyword_year_counts = collections.defaultdict(collections.Counter)
                yearly_document_counts = collections.Counter()
                keyword_cooccurrence_edges = collections.Counter()

                MAX_TERMS_PER_DOC = 12

                for _, work, _ in scored_works:
                    loc = work.get("primary_location") or {}
                    src = loc.get("source") or {}
                    j_title = src.get("display_name")
                    if j_title:
                        journal_stats[_clean_text_spaces(j_title)]["count"] += 1

                    yr = work.get("publication_year", 2020)
                    document_terms = extract_document_terms(work, max_terms=MAX_TERMS_PER_DOC)

                    for term in document_terms:
                        kw_freq[term] += 1

                    work_title = clean_title(work.get("title"))
                    work_cites = work.get("cited_by_count", 0)
                    authorships = work.get("authorships", [])

                    if authorships:
                        first_a = authorships[0]
                        a_obj = first_a.get("author", {})
                        a_id = a_obj.get("id")
                        a_name = a_obj.get("display_name")
                        if a_id and a_name:
                            author_stats[a_id]["name"] = _clean_text_spaces(a_name)
                            author_stats[a_id]["works"] += 1
                            author_stats[a_id]["cites"] += work_cites
                            author_stats[a_id]["years"].append(yr)
                            author_stats[a_id]["work_entries"].append(
                                {"title": work_title, "cites": work_cites}
                            )

                    work_countries = set()
                    for a in authorships:
                        for inst in a.get("institutions", []):
                            i_name = inst.get("display_name")
                            c_code = inst.get("country_code")
                            if i_name:
                                clean_inst = _clean_text_spaces(i_name)
                                inst_stats[clean_inst]["count"] += 1
                                inst_stats[clean_inst]["years"].append(yr)
                            if c_code:
                                work_countries.add(c_code)

                    for cc in work_countries:
                        country_stats[cc]["count"] += 1
                        country_stats[cc]["years"].append(yr)
                    if len(work_countries) > 1:
                        for cc in work_countries:
                            country_stats[cc]["intl"] += 1
                        for ca, cb in itertools.combinations(sorted(work_countries), 2):
                            country_coauth_edges[(ca, cb)] += 1

                    if yr:
                        yearly_document_counts[int(yr)] += 1
                        for term in document_terms:
                            keyword_year_counts[term][int(yr)] += 1

                    for ta, tb in itertools.combinations(sorted(document_terms), 2):
                        keyword_cooccurrence_edges[(ta, tb)] += 1

                inst_formatted = []
                for inst, data in sorted(
                    inst_stats.items(), key=lambda x: x[1]["count"], reverse=True
                )[:10]:
                    mean_yr = int(np.mean(data["years"])) if data["years"] else end_yr
                    inst_formatted.append({
                        "Institution Name": inst,
                        "Publication Count": data["count"],
                        "Mean Year": mean_yr,
                    })
                st.session_state["inst_distribution"] = inst_formatted

                country_formatted = []
                for c_code, data in sorted(
                    country_stats.items(), key=lambda x: x[1]["count"], reverse=True
                )[:10]:
                    mean_yr = int(np.mean(data["years"])) if data["years"] else end_yr
                    intl_pct = round(100 * data["intl"] / max(1, data["count"]), 1)
                    country_formatted.append({
                        "Country Code": c_code,
                        "Papers": data["count"],
                        "Mean Year": mean_yr,
                        "Intl Co-Auth %": intl_pct,
                    })
                st.session_state["country_distribution"] = country_formatted
                st.session_state["country_coauth_edges"] = dict(country_coauth_edges)

                journal_formatted = []
                for j_title, data in sorted(
                    journal_stats.items(), key=lambda x: x[1]["count"], reverse=True
                )[:10]:
                    journal_formatted.append({
                        "Journal / Source Title": j_title,
                        "Article Count": data["count"],
                    })
                st.session_state["journal_distribution"] = journal_formatted

                status.write("🔠 Computing keyword centrality…")
                keyword_graph_for_metrics = nx.Graph()
                for (ta, tb), weight in keyword_cooccurrence_edges.items():
                    if weight > 0:
                        keyword_graph_for_metrics.add_edge(ta, tb, weight=weight)
                keyword_betweenness = (
                    safe_betweenness(keyword_graph_for_metrics)
                    if keyword_graph_for_metrics.number_of_nodes() else {}
                )
                keyword_table_data = []
                for kw, freq in kw_freq.most_common(15):
                    keyword_table_data.append({
                        "Keyword": kw,
                        "Frequency": freq,
                        "Betweenness Centrality": round(
                            float(keyword_betweenness.get(kw, 0.0)), 5
                        ),
                    })
                st.session_state["keyword_centrality_data"] = keyword_table_data

                filtered_author_stats = {
                    a_id: data for a_id, data in author_stats.items()
                    if data["works"] >= 1
                }
                aca_formatted = []
                for a_id, stats in sorted(
                    filtered_author_stats.items(), key=lambda x: x[1]["cites"], reverse=True
                )[:15]:
                    yrs = stats["years"]
                    span = f"{min(yrs)}-{max(yrs)}" if yrs else "N/A"
                    top_w_title = "N/A"
                    if stats["work_entries"]:
                        best_work = max(stats["work_entries"], key=lambda x: x["cites"])
                        top_w_title = best_work["title"]
                    cites_per_work = round(stats["cites"] / max(1, stats["works"]), 1)
                    aca_formatted.append({
                        "Author Name": stats["name"],
                        "Works in Corpus (first-author)": stats["works"],
                        "Corpus Citation Sum": stats["cites"],
                        "Cites / Work": cites_per_work,
                        "Active": span,
                        "Top Work": top_w_title,
                    })
                st.session_state["aca_distribution"] = aca_formatted

                reference_frequency = collections.Counter()
                for corpus_work in results:
                    for reference_id in corpus_work.get("referenced_works", []) or []:
                        if reference_id:
                            reference_frequency[reference_id] += 1
                ranked_reference_ids = [
                    ref_id for ref_id, _ in reference_frequency.most_common(reference_metadata_cap)
                ]

                status.write(
                    f"📚 Retrieving metadata for up to {len(ranked_reference_ids)} cited works…"
                )
                reference_metadata, ref_requested, ref_errors = fetch_reference_metadata(
                    tuple(ranked_reference_ids)
                )
                if ref_errors:
                    total_batches = (ref_requested + 19) // 20
                    st.warning(
                        f"Reference metadata fetch had {len(ref_errors)} failed batch(es) "
                        f"out of {total_batches}. ACA and reference-burst tables are "
                        f"based on the {len(reference_metadata)} works that were "
                        f"successfully retrieved. First failure: {ref_errors[0]}"
                    )

                ACA_GRAPH, true_aca_data, _ = build_author_cocitation_analysis(
                    results, reference_metadata
                )
                st.session_state["true_aca_data"] = true_aca_data
                st.session_state["aca_reference_coverage"] = {
                    "requested": ref_requested,
                    "retrieved": len(reference_metadata),
                    "fetch_errors": len(ref_errors),
                }
                st.session_state["aca_graph"] = ACA_GRAPH

                status.write("🎨 Detecting communities + silhouette…")
                G_undirected = G.to_undirected()
                communities = []
                for component in nx.connected_components(G_undirected):
                    if len(component) < 2:
                        continue
                    cg = G_undirected.subgraph(component).copy()
                    try:
                        detected = list(nx.community.greedy_modularity_communities(cg))
                        communities.extend([set(c) for c in detected if c])
                    except Exception as e:
                        st.warning(f"Community detection skipped a component: {e}")

                silhouette_results = calculate_network_silhouette(G_undirected, communities)

                partition_communities = list(communities)
                clustered = set().union(*communities) if communities else set()
                isolate_count = 0
                for isolate in G_undirected.nodes():
                    if isolate not in clustered:
                        partition_communities.append({isolate})
                        isolate_count += 1
                modularity_q = None
                if G_undirected.number_of_edges() > 0 and partition_communities:
                    try:
                        modularity_q = float(nx.community.modularity(G_undirected, partition_communities))
                    except Exception as exc:
                        st.warning(f"Modularity could not be computed: {exc}")

                giant_stats = {}
                if G_undirected.number_of_nodes() > 0:
                    components = sorted(nx.connected_components(G_undirected),
                                        key=len, reverse=True)
                    giant_nodes = components[0]
                    GC = G_undirected.subgraph(giant_nodes).copy()
                    giant_stats = {
                        "nodes": GC.number_of_nodes(),
                        "edges": GC.number_of_edges(),
                        "density": round(nx.density(GC), 4),
                        "share_of_corpus_percent": round(
                            100 * GC.number_of_nodes() / G_undirected.number_of_nodes(), 2),
                    }
                    try:
                        gc_comms = list(nx.community.greedy_modularity_communities(GC))
                        if len(gc_comms) >= 2:
                            giant_stats["modularity_q_giant"] = round(
                                float(nx.community.modularity(GC, gc_comms)), 4)
                    except Exception:
                        giant_stats["modularity_q_giant"] = None
                    try:
                        gc_sil = calculate_network_silhouette(GC, gc_comms, max_component_size=500)
                        giant_stats["silhouette_giant"] = (
                            round(gc_sil["global_silhouette"], 4)
                            if gc_sil["global_silhouette"] is not None else None
                        )
                    except Exception:
                        giant_stats["silhouette_giant"] = None

                st.session_state["giant_component_stats"] = giant_stats

                work_by_id = {w.get("id"): w for w in results if w.get("id")}
                cluster_records = []
                for cluster_id, community in enumerate(communities):
                    member_works = [work_by_id[n] for n in community if n in work_by_id]
                    years_in_cluster = [
                        int(w["publication_year"]) for w in member_works
                        if w.get("publication_year")
                    ]
                    key_terms = get_cluster_label_terms(
                        member_works, kw_freq, len(results), top_n=3
                    )
                    label = " · ".join(key_terms) if key_terms else f"Cluster {cluster_id}"
                    leading_work = max(
                        member_works,
                        key=lambda w: next(
                            (sc for sc, sw, _ in scored_works if sw.get("id") == w.get("id")), 0
                        ),
                        default=None,
                    )
                    cluster_records.append({
                        "cluster_id": cluster_id,
                        "size": len(community),
                        "mean_year": round(float(np.mean(years_in_cluster)), 1) if years_in_cluster else None,
                        "min_year": min(years_in_cluster) if years_in_cluster else None,
                        "max_year": max(years_in_cluster) if years_in_cluster else None,
                        "label": label,
                        "key_terms": ", ".join(key_terms) if key_terms else "Not available",
                        "leading_work": clean_title(leading_work.get("title")) if leading_work else "N/A",
                        "silhouette": silhouette_results["cluster_scores"].get(cluster_id),
                    })

                cluster_records.sort(key=lambda x: x["size"], reverse=True)
                cluster_summary_data = []
                enhanced_clusters = []
                for record in cluster_records[:8]:
                    sil = record["silhouette"]
                    sil_disp = round(sil, 4) if sil is not None else "N/A"
                    cluster_summary_data.append({
                        "Cluster ID": f"Cluster #{record['cluster_id']}",
                        "Size": record["size"],
                        "Silhouette": sil_disp,
                        "Mean Year": record["mean_year"] if record["mean_year"] is not None else "N/A",
                        "Descriptive Label": record["label"],
                    })
                    enhanced_clusters.append({
                        "Cluster ID": f"Cluster #{record['cluster_id']}",
                        "Size": record["size"],
                        "Silhouette": sil_disp,
                        "Mean Year": record["mean_year"] if record["mean_year"] is not None else "N/A",
                        "Descriptive Label": record["label"],
                        "Key Terms": record["key_terms"],
                        "Active Years": (
                            f"{record['min_year']}-{record['max_year']}"
                            if record["min_year"] is not None else "N/A"
                        ),
                        "Leading Work": record["leading_work"],
                    })
                st.session_state["cluster_summary_data"] = cluster_summary_data
                st.session_state["enhanced_cluster_profiles"] = enhanced_clusters
                st.session_state["cluster_count_total"] = len(communities)

                unconnected_count = sum(
                    1 for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) == 0
                )
                st.session_state["net_stats"] = {
                    "nodes": G.number_of_nodes(),
                    "edges": G.number_of_edges(),
                    "clusters": len(communities),
                    "unconnected": unconnected_count,
                    "isolates": isolate_count,
                }

                sorted_by_year = sorted(
                    scored_works, key=lambda x: x[1].get("publication_year", 2000)
                )
                reading_path_data = []
                for _, (sc, work, _) in enumerate(sorted_by_year[:12], 1):
                    reading_path_data.append({
                        "Year": work.get("publication_year", "N/A"),
                        "Title": clean_title(work.get("title")),
                        "Authors": ", ".join(
                            [_clean_text_spaces(a.get("author", {}).get("display_name", ""))
                             for a in work.get("authorships", [])[:2]]
                        ) + " et al.",
                        "Score": sc,
                    })
                st.session_state["reading_path_data"] = reading_path_data

                def _short_excerpt(title_str, max_len=45):
                    s = clean_title(title_str)
                    if len(s) <= max_len:
                        return s
                    cut = s[:max_len].rsplit(" ", 1)[0]
                    return (cut or s[:max_len]) + "…"

                node_table_data = []
                work_id_to_title = {
                    w.get("id"): (clean_title(w.get("title")), w.get("publication_year", "N/A"))
                    for w in results if w.get("id")
                }
                for idx, (sc, work, role) in enumerate(scored_works[:30], 1):
                    w_id = work.get("id")
                    citing_nodes = []
                    cited_nodes = []
                    if w_id in G:
                        for p in G.predecessors(w_id):
                            if p in work_id_to_title:
                                t, y = work_id_to_title[p]
                                citing_nodes.append(f"{y}: {_short_excerpt(t, 38)}")
                        for s in G.successors(w_id):
                            if s in work_id_to_title:
                                t, y = work_id_to_title[s]
                                cited_nodes.append(f"{y}: {_short_excerpt(t, 38)}")
                    node_table_data.append({
                        "Rank": idx,
                        "Title": clean_title(work.get("title")),
                        "Year": work.get("publication_year", "N/A"),
                        "Score": sc,
                        "Cited By Nodes": "; ".join(citing_nodes[:3]) if citing_nodes else "-",
                        "Cites Nodes": "; ".join(cited_nodes[:3]) if cited_nodes else "-",
                    })
                st.session_state["node_table_data"] = node_table_data

                thematic_categories = {
                    "Indicators, Metrics & Ranking Terms": [],
                    "Methods, Models & Analytical Terms": [],
                    "Evaluation, Quality & Performance": [],
                    "Systems, Institutions & Contexts": [],
                    "Domain & Topical Keywords": [],
                }
                for kw, freq in kw_freq.items():
                    bucket = classify_thematic_keyword(kw)
                    thematic_categories[bucket].append((kw, freq))
                st.session_state["thematic_keywords"] = thematic_categories

                st.session_state["global_network_metrics"] = {
                    "modularity_q": round(modularity_q, 4) if modularity_q is not None else None,
                    "global_silhouette": (
                        round(silhouette_results["global_silhouette"], 4)
                        if silhouette_results["global_silhouette"] is not None else None
                    ),
                    "silhouette_evaluated_nodes": silhouette_results["evaluated_nodes"],
                    "silhouette_eligible_components": silhouette_results["eligible_components"],
                    "silhouette_skipped_components": silhouette_results["skipped_components"],
                    "silhouette_coverage_percent": round(
                        100 * silhouette_results["evaluated_nodes"] / max(1, G.number_of_nodes()), 2
                    ),
                    "isolate_fraction_percent": round(
                        100 * isolate_count / max(1, G.number_of_nodes()), 2
                    ),
                    "giant_component": giant_stats,
                }

                status.write("⏳ Detecting reference citation bursts…")
                all_years = list(range(start_yr, end_yr + 1))
                reference_citation_years = collections.defaultdict(collections.Counter)
                for citing_work in results:
                    cy = citing_work.get("publication_year")
                    if not cy:
                        continue
                    cy = int(cy)
                    for ref_id in set(citing_work.get("referenced_works", []) or []):
                        if ref_id in reference_metadata:
                            reference_citation_years[ref_id][cy] += 1

                ref_burst_data = []
                for ref_id, yearly_counts in reference_citation_years.items():
                    if sum(yearly_counts.values()) < 2:
                        continue
                    episodes = kleinberg_burst_episodes(
                        make_year_series(yearly_counts, all_years),
                        make_year_series(yearly_document_counts, all_years),
                        all_years,
                    )
                    meta = reference_metadata.get(ref_id, {})
                    for ep in episodes:
                        ref_burst_data.append({
                            "Reference / Publication": clean_title(meta.get("title") or "Untitled cited work"),
                            "Year": meta.get("publication_year", "N/A"),
                            "Burst Level": ep["Burst Level"],
                            "Begin": ep["Begin"],
                            "End": ep["End"],
                            "Events During Burst": ep["Events During Burst"],
                            "OpenAlex ID": ref_id,
                        })
                ref_burst_data.sort(
                    key=lambda it: (it["Burst Level"], it["Events During Burst"],
                                    it["End"] - it["Begin"] + 1),
                    reverse=True,
                )
                st.session_state["ref_burst_data"] = ref_burst_data[:15]

                # ================= FIGURES =================
                status.write("🎨 Rendering figures…")

                st.markdown("### 🕸️ 1. Document Citation Network")
                fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=130)
                pos1 = nx.spring_layout(G, seed=42)
                nx.draw(G, pos1, ax=ax1, node_size=60, node_color="#2563EB",
                        edge_color="#94A3B8", alpha=0.7, with_labels=False)
                ax1.set_title(
                    f"Citation Network (Nodes: {G.number_of_nodes()}, "
                    f"Edges: {G.number_of_edges()})",
                    fontsize=12, fontweight="bold",
                )
                st.pyplot(fig1, clear_figure=False)
                st.session_state["fig_citation"] = fig1
                plt.close(fig1)

                st.markdown("### ⏳ 2. Historical Timeline")
                slices = collections.defaultdict(list)
                for _, work, _ in scored_works:
                    yr = work.get("publication_year")
                    if yr:
                        bin_start = start_yr + ((int(yr) - start_yr) // time_slice_years) * time_slice_years
                        bin_end = min(end_yr, bin_start + time_slice_years - 1)
                        label = f"{bin_start}" if bin_start == bin_end else f"{bin_start}-{bin_end}"
                        slices[label].append(work)

                timeline_data = []
                for label, w_list in sorted(slices.items()):
                    tot = sum(w.get("cited_by_count", 0) for w in w_list)
                    years_in_slice = [w.get("publication_year") for w in w_list if w.get("publication_year")]
                    n = len(w_list)
                    cpp = tot / n if n else 0
                    mean_age = (end_yr + 1 - np.mean(years_in_slice)) if years_in_slice else 1.0
                    cpy = (tot / n / max(mean_age, 0.5)) if n else 0
                    timeline_data.append({
                        "Time Slice": label,
                        "Publications": n,
                        "Total Citations (lifetime, raw)": tot,
                        "Mean Citations / Paper": round(cpp, 1),
                        "Mean Citations / Paper / Year": round(cpy, 2),
                    })
                st.session_state["timeline_data"] = timeline_data

                fig_timeline, ax_t1 = plt.subplots(figsize=(11, 4.8), dpi=130)
                sdf = pd.DataFrame(timeline_data)
                x = np.arange(len(sdf["Time Slice"]))
                width = 0.38
                ax_t1.bar(x - width / 2, sdf["Publications"], width,
                          label="Publications", color="#2563EB")
                ax_t1.set_ylabel("Publication Count", color="#2563EB",
                                 fontweight="bold", fontsize=10)
                ax_t1.tick_params(axis="y", labelcolor="#2563EB", labelsize=9)
                ax_t2 = ax_t1.twinx()
                ax_t2.bar(x + width / 2, sdf["Total Citations (lifetime, raw)"], width,
                          label="Total Citations", color="#10B981", alpha=0.8)
                max_cites = max(sdf["Total Citations (lifetime, raw)"]) if len(sdf) else 1
                ax_t2.set_ylim(0, max(max_cites * 1.15, 10))
                ax_t2.set_ylabel("Total Citations (lifetime, raw)", color="#10B981",
                                 fontweight="bold", fontsize=10)
                ax_t2.tick_params(axis="y", labelcolor="#10B981", labelsize=9)
                plt.xticks(x, sdf["Time Slice"], rotation=45, ha="right", fontsize=9)
                plt.title("Historical Timeline Slicing & Corpus Evolution",
                          fontsize=11, fontweight="bold", pad=15)
                ax_t1.grid(True, linestyle="--", alpha=0.4)
                fig_timeline.subplots_adjust(right=0.88)
                plt.tight_layout()
                st.pyplot(fig_timeline, clear_figure=False)
                st.session_state["fig_timeline"] = fig_timeline
                plt.close(fig_timeline)
                st.dataframe(sdf)
                if recent_added_count > 0:
                    st.caption(
                        f"Note: {recent_added_count} works were added by the recent-work boost. "
                        "Publication counts for recent slices may be inflated relative to older slices."
                    )

                st.markdown("### 🎨 3. Thematic Cluster Analysis")
                st.dataframe(pd.DataFrame(cluster_summary_data))
                if silhouette_results["global_silhouette"] is not None:
                    st.caption(
                        f"Network silhouette: {silhouette_results['global_silhouette']:.4f} "
                        f"on {silhouette_results['evaluated_nodes']} of {G.number_of_nodes()} nodes "
                        f"({100 * silhouette_results['evaluated_nodes'] / max(1, G.number_of_nodes()):.1f}% coverage; "
                        f"{silhouette_results['skipped_components']} oversized components skipped)."
                    )

                fig_cluster, ax_cluster = plt.subplots(figsize=(10, 5.5), dpi=130)
                if cluster_summary_data:
                    cdf_plot = pd.DataFrame(cluster_summary_data)
                    y_cl = np.arange(len(cdf_plot))
                    cl_colors = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
                                 "#EC4899", "#06B6D4", "#F97316"]
                    ax_cluster.barh(
                        y_cl, cdf_plot["Size"],
                        color=[cl_colors[i % len(cl_colors)] for i in range(len(cdf_plot))],
                        alpha=0.88,
                    )
                    ax_cluster.set_yticks(y_cl)
                    ax_cluster.set_yticklabels(
                        [f"{r['Cluster ID']} ({r['Descriptive Label']})" for _, r in cdf_plot.iterrows()],
                        fontsize=8.5, fontweight="bold",
                    )
                    ax_cluster.invert_yaxis()
                    ax_cluster.set_xlabel("Cluster Size (Publications)", fontweight="bold")
                    ax_cluster.grid(True, linestyle="--", alpha=0.4, axis="x")
                else:
                    ax_cluster.text(0.5, 0.5, "No multi-node clusters detected",
                                    ha="center", va="center", transform=ax_cluster.transAxes)
                ax_cluster.set_title("Thematic Cluster Sizes", fontsize=11, fontweight="bold", pad=12)
                plt.tight_layout()
                st.pyplot(fig_cluster, clear_figure=False)
                st.session_state["fig_cluster"] = fig_cluster
                plt.close(fig_cluster)

                st.markdown("### 🔠 4. Keyword Co-Occurrence & Bursts")
                st.dataframe(pd.DataFrame(keyword_table_data))

                fig_kw_net, ax_kwn = plt.subplots(figsize=(9, 6.5), dpi=130)
                KWG = nx.Graph()
                top_kw_set = {item["Keyword"] for item in keyword_table_data}
                for (ta, tb), w in keyword_cooccurrence_edges.items():
                    if w >= 3 and ta in top_kw_set and tb in top_kw_set:
                        KWG.add_edge(ta, tb, weight=w)
                if KWG.number_of_nodes() == 0:
                    for item in keyword_table_data[:14]:
                        KWG.add_node(item["Keyword"])
                if KWG.number_of_nodes() > 0:
                    pos_kw = nx.spring_layout(KWG, seed=42, k=0.65, iterations=120, weight="weight")
                    sizes = [max(300, kw_freq.get(n, 1) * 3.0) for n in KWG.nodes]
                    widths = [0.6 + 0.25 * KWG[u][v].get("weight", 1) for u, v in KWG.edges()]
                    nx.draw_networkx_nodes(KWG, pos_kw, node_size=sizes,
                                           node_color="#F59E0B", alpha=0.9, ax=ax_kwn)
                    if KWG.number_of_edges() > 0:
                        nx.draw_networkx_edges(KWG, pos_kw, width=widths,
                                               edge_color="#94A3B8", alpha=0.6, ax=ax_kwn)
                    nx.draw_networkx_labels(KWG, pos_kw, font_size=9, font_weight="bold",
                                            font_color="#1E293B", ax=ax_kwn)
                ax_kwn.set_title("Keyword Co-Occurrence Network (edges = co-occurrence ≥ 3)",
                                 fontsize=11, fontweight="bold", pad=12)
                ax_kwn.margins(0.08)
                ax_kwn.axis("off")
                plt.tight_layout()
                st.pyplot(fig_kw_net, clear_figure=False)
                st.session_state["fig_kw_net"] = fig_kw_net
                plt.close(fig_kw_net)

                burst_results = []
                for kw, _ in kw_freq.most_common(25):
                    yearly_counts = keyword_year_counts.get(kw, collections.Counter())
                    episodes = kleinberg_burst_episodes(
                        make_year_series(yearly_counts, all_years),
                        make_year_series(yearly_document_counts, all_years),
                        all_years,
                    )
                    for ep in episodes:
                        burst_results.append({
                            "Keyword": kw,
                            "Frequency": int(sum(yearly_counts.values())),
                            "Burst Level": ep["Burst Level"],
                            "Begin": ep["Begin"],
                            "End": ep["End"],
                            "Documents During Burst": ep["Events During Burst"],
                        })
                burst_results.sort(
                    key=lambda it: (it["Burst Level"], it["Documents During Burst"],
                                    it["End"] - it["Begin"] + 1),
                    reverse=True,
                )
                st.session_state["burst_results"] = burst_results[:15]
                st.session_state["burst_scope"] = {
                    "keywords_evaluated": min(25, len(kw_freq)),
                    "total_keywords": len(kw_freq),
                    "episodes_found": len(burst_results),
                    "window_years": len(all_years),
                    "parameters": {"s": 2.0, "gamma": 1.0},
                }

                max_burst_level = max((b["Burst Level"] for b in burst_results), default=0)
                if len(burst_results) >= 2 and max_burst_level >= 2:
                    fig_burst, ax_b = plt.subplots(figsize=(10, 6), dpi=130)
                    bdf = pd.DataFrame(burst_results[:15])
                    y_pos = np.arange(len(bdf))
                    ax_b.barh(y_pos, bdf["Burst Level"], color="#EF4444", alpha=0.85)
                    ax_b.set_yticks(y_pos)
                    ax_b.set_yticklabels(bdf["Keyword"], fontweight="bold", fontsize=9)
                    ax_b.invert_yaxis()
                    ax_b.set_xlabel("Kleinberg Burst Level", fontweight="bold")
                    ax_b.set_title("Detected Keyword Bursts (Kleinberg Model)",
                                   fontsize=12, fontweight="bold", pad=15)
                    ax_b.grid(True, linestyle="--", alpha=0.5, axis="x")
                    plt.tight_layout()
                    st.pyplot(fig_burst, clear_figure=False)
                    st.session_state["fig_burst"] = fig_burst
                    plt.close(fig_burst)
                else:
                    st.info(
                        f"Only {len(burst_results)} keyword burst(s) detected "
                        f"(max level = {max_burst_level}); chart suppressed because "
                        "burst level ≥ 2 was not found. This is a power limitation, not "
                        "evidence of absence. See report §7."
                    )
                    st.dataframe(pd.DataFrame(burst_results))

                st.markdown("### 👥 5. First-Author Table & Author Co-Citation (ACA)")
                st.dataframe(pd.DataFrame(st.session_state["aca_distribution"]))
                st.caption(
                    "Table above is corpus-internal citation sums for first authors "
                    "(sorted by sum, not productivity). See 'Cites / Work' for per-paper impact. "
                    f"ACA used {len(reference_metadata)} retrieved cited-work records "
                    f"of {ref_requested} selected IDs."
                )
                st.markdown("**Author Co-Citation Analysis (ACA):**")
                st.dataframe(pd.DataFrame(true_aca_data))

                fig_auth_net, ax_an = plt.subplots(figsize=(8.5, 6), dpi=130)
                top_aca_ids = [row["OpenAlex Author ID"] for row in true_aca_data[:15]]
                ANG = ACA_GRAPH.subgraph(top_aca_ids).copy() if top_aca_ids else nx.Graph()
                if ANG.number_of_nodes() > 0:
                    labels_aca = {n: ANG.nodes[n].get("name", n) for n in ANG.nodes()}
                    pos_an = nx.spring_layout(ANG, seed=42, weight="weight", k=1.1, iterations=120)
                    nx.draw_networkx_nodes(ANG, pos_an, node_size=420,
                                           node_color="#8B5CF6", alpha=0.9, ax=ax_an)
                    widths = [0.8 + 0.35 * ANG[u][v].get("weight", 1) for u, v in ANG.edges()]
                    nx.draw_networkx_edges(ANG, pos_an, width=widths,
                                           edge_color="#94A3B8", alpha=0.75, ax=ax_an)
                    nx.draw_networkx_labels(ANG, pos_an, labels=labels_aca, font_size=7.5,
                                            font_weight="bold", font_color="#1E293B", ax=ax_an)
                else:
                    ax_an.text(0.5, 0.5,
                               "Insufficient retrieved reference metadata to construct an ACA network.",
                               ha="center", va="center", transform=ax_an.transAxes)
                ax_an.set_title("Author Co-Citation Network (ACA)", fontsize=11,
                                fontweight="bold", pad=12)
                ax_an.margins(0.15)
                ax_an.axis("off")
                plt.tight_layout()
                st.pyplot(fig_auth_net, clear_figure=False)
                st.session_state["fig_auth_net"] = fig_auth_net
                plt.close(fig_auth_net)

                st.markdown("### 🏛️ 6. Institutions & Countries")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Top Research Institutions**")
                    st.dataframe(pd.DataFrame(st.session_state["inst_distribution"]))
                with c2:
                    st.markdown("**Top Countries / Regions**")
                    st.dataframe(pd.DataFrame(st.session_state["country_distribution"]))

                fig_cnt_net, ax_cn = plt.subplots(figsize=(8, 5), dpi=130)
                cdf = pd.DataFrame(st.session_state["country_distribution"])
                if not cdf.empty:
                    y_c = np.arange(len(cdf))
                    ax_cn.barh(y_c, cdf["Papers"], color="#2563EB", alpha=0.85)
                    ax_cn.set_yticks(y_c)
                    ax_cn.set_yticklabels(cdf["Country Code"], fontweight="bold", fontsize=9)
                    ax_cn.invert_yaxis()
                    ax_cn.set_xlabel("Papers", fontweight="bold")
                    ax_cn.grid(True, linestyle="--", alpha=0.4, axis="x")
                else:
                    ax_cn.text(0.5, 0.5, "No country data", ha="center",
                               va="center", transform=ax_cn.transAxes)
                ax_cn.set_title("Country Publication Distribution",
                                fontsize=11, fontweight="bold", pad=12)
                plt.tight_layout()
                st.pyplot(fig_cnt_net, clear_figure=False)
                st.session_state["fig_cnt_net"] = fig_cnt_net
                plt.close(fig_cnt_net)

                st.markdown("### 📖 7. Publication Source Distribution")
                st.dataframe(pd.DataFrame(st.session_state["journal_distribution"]))
                fig_jnl_net, ax_jn = plt.subplots(figsize=(9, 5), dpi=130)
                jdf = pd.DataFrame(st.session_state["journal_distribution"])
                if not jdf.empty:
                    y_j = np.arange(len(jdf))
                    ax_jn.barh(y_j, jdf["Article Count"], color="#10B981", alpha=0.85)
                    ax_jn.set_yticks(y_j)
                    ax_jn.set_yticklabels(
                        [t[:45] + ("…" if len(t) > 45 else "") for t in jdf["Journal / Source Title"]],
                        fontsize=8,
                    )
                    ax_jn.invert_yaxis()
                    ax_jn.set_xlabel("Article Count", fontweight="bold")
                    ax_jn.grid(True, linestyle="--", alpha=0.4, axis="x")
                else:
                    ax_jn.text(0.5, 0.5, "No journal data", ha="center",
                               va="center", transform=ax_jn.transAxes)
                ax_jn.set_title("Journal / Source Distribution",
                                fontsize=11, fontweight="bold", pad=12)
                plt.tight_layout()
                st.pyplot(fig_jnl_net, clear_figure=False)
                st.session_state["fig_jnl_net"] = fig_jnl_net
                plt.close(fig_jnl_net)

                for score, work, role in scored_works[:20]:
                    title = clean_title(work.get("title"))
                    cites = work.get("cited_by_count", 0)
                    py = work.get("publication_year", "N/A")
                    wid = work.get("id", "N/A")
                    doi = work.get("doi", "N/A")
                    with st.expander(
                        f"[Score: {score} | Role: {role}] {title} — ({cites} citations | {py})"
                    ):
                        st.write(f"**Publication Year:** {py}")
                        st.write(f"**Total Global Citations:** {cites}")
                        st.write(f"**Composite Score (exploratory):** {score}")
                        st.write(f"**Assigned Role(s):** {role}")
                        st.write(f"**OpenAlex ID:** `{wid}`")
                        if doi:
                            st.write(f"**DOI Link:** {doi}")

                status.update(label="Analysis complete ✅", state="complete")

            except Exception as e:
                status.update(label="Analysis failed", state="error")
                st.error(f"An error occurred during analysis: {e}")
                st.exception(e)


# ================= REPORT EXPORT =================
if "scored_works" in st.session_state and st.session_state["scored_works"]:
    st.markdown("---")
    st.subheader("📥 Download Research Report (Word .docx)")

    if st.button("Generate the Network Analysis Report (.docx)"):
        scored_works = st.session_state["scored_works"]
        query_text = st.session_state.get("search_query", "N/A")
        oa_query_text = st.session_state.get("openalex_query_string", query_text)
        retrieved_at = st.session_state.get("retrieval_timestamp", "N/A")
        rp = st.session_state.get("run_params", {})
        r_match_mode = rp.get("match_mode", match_mode)
        r_token_logic = rp.get("token_logic", token_logic)
        r_exact_phrase = rp.get("exact_phrase_mode", exact_phrase_mode)
        r_search_scope = rp.get("search_scope", search_scope)
        r_paratext = rp.get("filter_paratext", filter_paratext)
        r_start_yr = rp.get("start_yr", pub_years[0])
        r_end_yr = rp.get("end_yr", pub_years[1])
        r_max_results = rp.get("max_results", max_results)
        r_node_cap = rp.get("node_cap", node_cap)
        r_ref_cap = rp.get("reference_metadata_cap", reference_metadata_cap)
        r_sort_option = rp.get("sort_option", sort_option)
        r_add_refs = rp.get("add_cited_refs", add_cited_refs)
        r_boost = rp.get("boost_latest", boost_latest)
        r_slice_yrs = rp.get("time_slice_years", time_slice_years)

        cov = st.session_state.get("coverage_stats", {})
        net_stats = st.session_state.get("net_stats", {})
        nm = st.session_state.get("global_network_metrics", {})
        gc = nm.get("giant_component", {}) or {}
        true_aca_data = st.session_state.get("true_aca_data", []) or []

        for a in true_aca_data:
            expected = round(a["Co-Citation Link Strength"] / max(1, a["Citing Corpus Documents"]), 2)
            if abs(float(a["Link / Citing Doc"]) - float(expected)) > 1e-6:
                raise RuntimeError(
                    f"ACA table integrity failure for {a['Co-Cited Author']}: "
                    f"Link/Doc={a['Link / Citing Doc']} != expected={expected}"
                )

        if query_text and query_text != "N/A":
            check_filter = make_relevance_filter(
                [p.strip() for p in query_text.split(";") if p.strip()],
                match_mode=r_match_mode,
                search_scope=r_search_scope,
                exact_phrase=r_exact_phrase,
                token_logic=r_token_logic,
            )
            for _, w, _ in scored_works:
                if r_paratext and not is_substantive_work(w):
                    raise RuntimeError(f"Non-substantive work slipped into corpus: {w.get('title')}")
                if check_filter is not None and not check_filter(w):
                    raise RuntimeError(f"Off-query work slipped into corpus: {w.get('title')}")

        def fig_to_bytes(fig_key, dpi=300):
            if fig_key not in st.session_state:
                return io.BytesIO()
            buf = io.BytesIO()
            try:
                st.session_state[fig_key].savefig(
                    buf, format="png", bbox_inches="tight", dpi=dpi
                )
                buf.seek(0)
            except Exception:
                return io.BytesIO()
            return buf

        img_io_1 = fig_to_bytes("fig_citation")
        img_io_timeline = fig_to_bytes("fig_timeline")
        img_io_cluster = fig_to_bytes("fig_cluster")
        img_io_kw = fig_to_bytes("fig_kw_net")
        img_io_burst = fig_to_bytes("fig_burst")
        img_io_auth = fig_to_bytes("fig_auth_net")
        img_io_cnt = fig_to_bytes("fig_cnt_net")
        img_io_jnl = fig_to_bytes("fig_jnl_net")

        doc = Document()
        doc.add_heading("NAGraph: Bibliometric & Citation Network Analysis Report", level=1)
        doc.add_paragraph(
            _clean_text_spaces(
                f"Query Keywords: {query_text} | "
                f"OpenAlex Query Expression: {oa_query_text} | "
                f"Data Retrieval Date: {retrieved_at} | "
                f"Total Analysed Publications: {len(scored_works)}. "
                "Methodology: OpenAlex retrieval, directed citation network, "
                "greedy-modularity community detection, shortest-path silhouette on eligible "
                "components, document-level keyword co-occurrence, Kleinberg-style burst "
                "detection, author co-citation analysis from retrieved cited-work metadata, "
                "first-author summaries, source/institution/country counts, multi-metric "
                "publication scoring. Two partitions are used: modularity Q is computed on the "
                "isolate-augmented partition (isolates as singletons), while silhouette is computed "
                "only on components with 3-500 nodes and excludes isolates. This convention is stated "
                "so that Q and silhouette are not misread as describing the same partition. "
                "Developed by Dr. Nibu A. George, Baselius College Kottayam (Autonomous)."
            )
        )

        doc.add_heading("1. Search Parameters", level=2)
        pt = doc.add_table(rows=1, cols=2)
        pt.style = "Table Grid"
        pt.rows[0].cells[0].text = "Parameter"
        pt.rows[0].cells[1].text = "Value"
        for k, v in [
            ("Keywords", query_text),
            ("OpenAlex query string", oa_query_text),
            ("Retrieval date (UTC)", retrieved_at),
            ("Matching mode (across lines)", r_match_mode),
            ("Within-line token logic", r_token_logic),
            ("Exact phrase mode", "Yes" if r_exact_phrase else "No"),
            ("Publication years", f"{r_start_yr} - {r_end_yr}"),
            ("Retrieved by", r_sort_option),
            ("Max works to retrieve (slider)", r_max_results),
            ("Node cap (slider)", r_node_cap),
            ("Reference metadata cap (slider)", r_ref_cap),
            ("Timeline slice window (years)", r_slice_yrs),
            ("Reference expansion", "Yes" if r_add_refs else "No"),
            ("Recent-work boost", "Yes" if r_boost else "No"),
            ("Relevance filter", cov.get("filter_description", "none")),
            ("Data source", "OpenAlex (https://openalex.org)"),
        ]:
            r = pt.add_row().cells
            r[0].text = k
            r[1].text = str(v)

        doc.add_heading("2. Data Coverage & Node Cap", level=2)
        oa_total = cov.get("openalex_total_matches")
        oa_str = f"{oa_total:,}" if isinstance(oa_total, int) else "N/A (API count unavailable)"
        doc.add_paragraph(
            f"OpenAlex total matches (pre-filter): {oa_str}\n"
            f"{cov.get('base', 0)} works retrieved by search (post-dedup); "
            f"{cov.get('recent', 0)} added by recent boost; "
            f"{cov.get('one_hop', 0)} added by one-hop expansion. "
            f"{cov.get('total_candidates', 0)} candidates; network limited to "
            f"{len(scored_works)} nodes; {cov.get('dropped', 0)} excluded."
        )
        doc.add_paragraph(
            "Note: corpus composition depends on the retrieval parameters shown in §1. "
            "Different slider values, query expressions, or match modes will produce different "
            "top-ranked works. Comparisons across runs should verify parameter equivalence."
        )

        doc.add_heading("3. Topic Overview", level=2)
        top_score, top_work, _ = scored_works[0] if scored_works else (0.0, {}, "")
        tw = clean_title(top_work.get("title")) if top_work else "N/A"
        tc = top_work.get("cited_by_count", 0) if top_work else 0
        most_cited = max(scored_works, key=lambda x: x[1].get("cited_by_count", 0)) if scored_works else (0, {}, "")
        mc_title = clean_title(most_cited[1].get("title")) if most_cited[1] else "N/A"
        mc_cites = most_cited[1].get("cited_by_count", 0) if most_cited[1] else 0
        doc.add_paragraph(
            f"Network contains {len(scored_works)} works and {net_stats.get('edges', 0)} "
            f"citation links, published between {r_start_yr} and {r_end_yr}. "
            f"Highest-scoring work under the composite metric (score {top_score}; "
            f"see §11 for weight sensitivity): \"{tw}\", cited {tc} times overall. "
            f"Most-cited work in the corpus: \"{mc_title}\", cited {mc_cites} times overall."
        )

        doc.add_heading("4. Network Statistics", level=2)
        st_tbl = doc.add_table(rows=1, cols=2)
        st_tbl.style = "Table Grid"
        st_tbl.rows[0].cells[0].text = "Measure"
        st_tbl.rows[0].cells[1].text = "Value"
        for k, v in [
            ("Works (nodes)", net_stats.get("nodes", len(scored_works))),
            ("Citation links (edges)", net_stats.get("edges", 0)),
            ("Thematic clusters", net_stats.get("clusters", 0)),
            ("Global Modularity (Q)", nm.get("modularity_q", "Not calculable")),
            ("Network Silhouette", nm.get("global_silhouette", "Not calculable")),
            ("Silhouette evaluated nodes", nm.get("silhouette_evaluated_nodes", 0)),
            ("Silhouette coverage (%)", nm.get("silhouette_coverage_percent", 0)),
            ("Oversized components skipped", nm.get("silhouette_skipped_components", 0)),
            ("Isolated nodes (fraction %)", nm.get("isolate_fraction_percent", 0)),
            ("Unconnected works", net_stats.get("unconnected", 0)),
            ("Giant component nodes", gc.get("nodes", "N/A")),
            ("Giant component edges", gc.get("edges", "N/A")),
            ("Giant component density", gc.get("density", "N/A")),
            ("Giant component share of corpus (%)", gc.get("share_of_corpus_percent", "N/A")),
            ("Modularity Q (giant component only)", gc.get("modularity_q_giant", "N/A")),
            ("Silhouette (giant component only)", gc.get("silhouette_giant", "N/A")),
        ]:
            r = st_tbl.add_row().cells
            r[0].text = k
            r[1].text = str(v)

        doc.add_paragraph(
            f"Note on modularity: {nm.get('isolate_fraction_percent', 0)}% of nodes are "
            "isolates (no internal citation link). The whole-graph modularity "
            f"(Q = {nm.get('modularity_q', 'N/A')}) is therefore inflated by isolates and "
            "is reported only for completeness. The methodologically meaningful value is "
            f"the giant-component modularity (Q = {gc.get('modularity_q_giant', 'N/A')}), "
            f"computed on {gc.get('nodes', 0)} nodes and {gc.get('edges', 0)} edges "
            f"({gc.get('share_of_corpus_percent', 0)}% of the corpus). "
            "The same convention applies to the silhouette metric, which is additionally "
            "restricted to eligible components of 3-500 nodes."
        )

        sil_giant_val = gc.get("silhouette_giant")
        if isinstance(sil_giant_val, (int, float)):
            if sil_giant_val > 0.5:
                sil_band = "strong cluster structure (>0.5)"
            elif sil_giant_val >= 0.25:
                sil_band = "weak-to-moderate cluster structure (0.25-0.5)"
            else:
                sil_band = "weak or overlapping cluster structure (<0.25)"
            doc.add_paragraph(
                f"Interpretation of silhouette: values 0.25-0.5 indicate weak-to-moderate "
                f"cluster structure, >0.5 indicates strong structure. The giant-component "
                f"silhouette ({sil_giant_val}) falls in the {sil_band}, indicating that "
                f"the connected citation core is only partially partitionable into distinct communities."
            )
        else:
            doc.add_paragraph(
                "Interpretation of silhouette: values 0.25-0.5 indicate weak-to-moderate "
                "cluster structure and >0.5 indicates strong structure. Here, silhouette is "
                "not defined (None) because no connected component meets the minimum "
                "3-node threshold required for shortest-path silhouette evaluation."
            )

        if isinstance(gc.get("nodes"), int) and gc.get("nodes", 0) < 50:
            doc.add_paragraph(
                f"Caveat: with a giant component of only {gc.get('nodes', 0)} nodes "
                f"({gc.get('share_of_corpus_percent', 0)}% of the corpus), the "
                "giant-component metrics describe the largest weakly-connected "
                "citation chain, not a macroscopic connected core. The isolate "
                "fraction is itself a key structural finding regarding internal "
                "citation connectivity within the retrieved set."
            )

        doc.add_heading("Document Citation Network Map", level=3)
        if img_io_1.getbuffer().nbytes > 0:
            doc.add_picture(img_io_1, width=docx.shared.Inches(5.5))

        doc.add_heading("5. Historical Timeline", level=2)
        if img_io_timeline.getbuffer().nbytes > 0:
            doc.add_picture(img_io_timeline, width=docx.shared.Inches(5.5))
        timeline_cols = [
            "Time Slice", "Publications", "Total Citations (lifetime)",
            "Mean Cites/Paper", "Mean Cites/Paper/Year",
        ]
        tt = doc.add_table(rows=1, cols=len(timeline_cols))
        tt.style = "Table Grid"
        tt.autofit = False
        for i, h in enumerate(timeline_cols):
            tt.rows[0].cells[i].text = h
        for item in st.session_state.get("timeline_data", []):
            r = tt.add_row().cells
            r[0].text = str(item["Time Slice"])
            r[1].text = str(item["Publications"])
            r[2].text = str(item["Total Citations (lifetime, raw)"])
            r[3].text = str(item["Mean Citations / Paper"])
            r[4].text = str(item["Mean Citations / Paper / Year"])
        doc.add_paragraph(
            "Important: 'Total Citations (lifetime)' is a raw cumulative count and "
            "declines mechanically with recency because recent papers have had less "
            "time to accumulate citations. The normalized columns "
            "'Mean Cites/Paper' and 'Mean Cites/Paper/Year' control for this and are "
            "the values that should be interpreted as an impact signal. No field-"
            "normalization (e.g., MNCS) has been applied, so cross-slice comparisons "
            "remain domain-dependent."
        )
        if cov.get("recent", 0) > 0:
            doc.add_paragraph(
                f"Note: {cov.get('recent', 0)} works were added by the recent-work boost. "
                "Publication counts for recent slices may be inflated relative to older slices."
            )

        doc.add_heading("6. Cluster Profiles", level=2)
        if img_io_cluster.getbuffer().nbytes > 0:
            doc.add_picture(img_io_cluster, width=docx.shared.Inches(5.5))
        ct = doc.add_table(rows=1, cols=6)
        ct.style = "Table Grid"
        for i, h in enumerate(["Cluster", "Label", "Size", "Key Terms", "Active Years", "Leading Work"]):
            ct.rows[0].cells[i].text = h
        for c in st.session_state.get("enhanced_cluster_profiles", []):
            r = ct.add_row().cells
            r[0].text = str(c["Cluster ID"])
            r[1].text = str(c["Descriptive Label"])
            r[2].text = str(c["Size"])
            r[3].text = str(c["Key Terms"])
            r[4].text = str(c["Active Years"])
            r[5].text = str(c["Leading Work"])
        doc.add_paragraph(
            "Cluster labels use TF-IDF weighted term frequency: terms frequent within a "
            "cluster but rare corpus-wide are prioritised to highlight sub-topic distinction."
        )

        doc.add_heading("Chronological Reading Path", level=3)
        rpt = doc.add_table(rows=1, cols=4)
        rpt.style = "Table Grid"
        for i, h in enumerate(["Year", "Title", "Authors", "Score"]):
            rpt.rows[0].cells[i].text = h
        for rp in st.session_state.get("reading_path_data", []):
            r = rpt.add_row().cells
            r[0].text = str(rp["Year"])
            r[1].text = str(rp["Title"])
            r[2].text = str(rp["Authors"])
            r[3].text = str(rp["Score"])

        doc.add_heading("7. Keyword Co-Occurrence & Thematic Categorization", level=2)
        if img_io_kw.getbuffer().nbytes > 0:
            doc.add_picture(img_io_kw, width=docx.shared.Inches(5.5))
        for category, items in st.session_state.get("thematic_keywords", {}).items():
            if items:
                doc.add_heading(f"Category: {category}", level=3)
                ctbl = doc.add_table(rows=1, cols=2)
                ctbl.style = "Table Grid"
                ctbl.rows[0].cells[0].text = "Keyword"
                ctbl.rows[0].cells[1].text = "Frequency"
                for kw_name, kw_val in sorted(items, key=lambda x: x[1], reverse=True)[:5]:
                    r = ctbl.add_row().cells
                    r[0].text = str(kw_name)
                    r[1].text = str(kw_val)

        doc.add_heading("Keyword Bursts", level=3)
        if "fig_burst" in st.session_state and img_io_burst.getbuffer().nbytes > 0:
            doc.add_picture(img_io_burst, width=docx.shared.Inches(5.5))
        bt = doc.add_table(rows=1, cols=5)
        bt.style = "Table Grid"
        for i, h in enumerate(["Keyword", "Frequency", "Level", "Begin", "End"]):
            bt.rows[0].cells[i].text = h
        for b in st.session_state.get("burst_results", []):
            r = bt.add_row().cells
            r[0].text = str(b["Keyword"])
            r[1].text = str(b["Frequency"])
            r[2].text = str(b["Burst Level"])
            r[3].text = str(b["Begin"])
            r[4].text = str(b["End"])

        bs = st.session_state.get("burst_scope", {})
        doc.add_paragraph(
            f"Burst scope and limitation: Kleinberg burst detection "
            f"(s = {bs.get('parameters', {}).get('s', 2.0)}, "
            f"gamma = {bs.get('parameters', {}).get('gamma', 1.0)}) was applied to the "
            f"top {bs.get('keywords_evaluated', 0)} keywords by corpus frequency "
            f"(of {bs.get('total_keywords', 0)} distinct keywords) over a "
            f"{bs.get('window_years', 0)}-year window. "
            f"{bs.get('episodes_found', 0)} burst episode(s) were detected. "
            "This is a power limitation, not evidence of absence: with annual keyword "
            "document counts mostly in the range 0-3, the Poisson emission model has "
            "insufficient signal to distinguish a burst from noise. Burst results "
            "should be interpreted as indicative rather than definitive."
        )

        ref_bursts = st.session_state.get("ref_burst_data", [])
        if ref_bursts:
            doc.add_heading("Reference Citation Bursts", level=3)
            rbt = doc.add_table(rows=1, cols=5)
            rbt.style = "Table Grid"
            for i, h in enumerate(["Title", "Year", "Level", "Begin", "End"]):
                rbt.rows[0].cells[i].text = h
            for rb in ref_bursts:
                r = rbt.add_row().cells
                r[0].text = str(rb["Reference / Publication"])
                r[1].text = str(rb["Year"])
                r[2].text = str(rb["Burst Level"])
                r[3].text = str(rb["Begin"])
                r[4].text = str(rb["End"])

            earliest = min(ref_bursts, key=lambda x: x["Begin"])
            latest = max(ref_bursts, key=lambda x: x["End"])
            doc.add_paragraph(
                f"Interpretation: the earliest detected reference burst begins in "
                f"{earliest['Begin']} on \"{earliest['Reference / Publication'][:80]}\", "
                f"and the most recent burst extends to {latest['End']} on "
                f"\"{latest['Reference / Publication'][:80]}\". Together these mark the "
                f"temporal evolution of the field's citation base."
            )

        doc.add_heading("8. First-Author Corpus Citation Sums", level=2)
        at = doc.add_table(rows=1, cols=6)
        at.style = "Table Grid"
        at.autofit = False
        for i, h in enumerate(["#", "Author", "Works (1st-author, corpus)",
                               "Corpus Citation Sum", "Cites / Work", "Active / Top Work"]):
            at.rows[0].cells[i].text = h
        for i, a in enumerate(st.session_state.get("aca_distribution", []), 1):
            r = at.add_row().cells
            r[0].text = str(i)
            r[1].text = str(a["Author Name"])
            r[2].text = str(a["Works in Corpus (first-author)"])
            r[3].text = str(a["Corpus Citation Sum"])
            r[4].text = str(a["Cites / Work"])
            r[5].text = f"{a['Active']} | {a['Top Work']}"
        doc.add_paragraph(
            "Corpus Citation Sum = sum of cited_by_count for corpus works in which the "
            "person is first author. This is a corpus-internal measure and is NOT the "
            "author's full-career citation total."
        )

        doc.add_heading("Author Co-Citation Analysis (ACA)", level=2)
        if img_io_auth.getbuffer().nbytes > 0:
            doc.add_picture(img_io_auth, width=docx.shared.Inches(5.5))
        tat = doc.add_table(rows=1, cols=6)
        tat.style = "Table Grid"
        tat.autofit = False
        tat.alignment = WD_TABLE_ALIGNMENT.LEFT
        for i, h in enumerate(["Rank", "Co-Cited Author", "Citing Docs", "Link Strength", "Link / Doc", "Betweenness"]):
            tat.rows[0].cells[i].text = h
        for a in true_aca_data:
            r = tat.add_row().cells
            r[0].text = str(a["Rank"])
            r[1].text = str(a["Co-Cited Author"])
            r[2].text = str(a["Citing Corpus Documents"])
            r[3].text = str(a["Co-Citation Link Strength"])
            r[4].text = f"{float(a['Link / Citing Doc']):.2f}"
            r[5].text = f"{float(a['Betweenness Centrality']):.5f}"
        ac = st.session_state.get("aca_reference_coverage", {})
        doc.add_paragraph(
            f"ACA reference metadata coverage: {ac.get('retrieved', 0)} "
            f"of {ac.get('requested', 0)} selected cited works retrieved "
            f"({ac.get('fetch_errors', 0)} fetch batch failures). "
            "Co-citation counts are based on co-occurrence of cited authors in the "
            "reference lists of corpus publications."
        )

        if true_aca_data:
            top = true_aca_data[0]
            top_first_sums = {r["Author Name"]: r["Corpus Citation Sum"]
                              for r in st.session_state.get("aca_distribution", [])}
            aca_by_name = {r["Co-Cited Author"]: r for r in true_aca_data}
            low_btw_example = None
            for name, _ in sorted(top_first_sums.items(), key=lambda x: -x[1]):
                if name in aca_by_name and aca_by_name[name]["Betweenness Centrality"] < 0.03:
                    low_btw_example = (name, aca_by_name[name]["Betweenness Centrality"])
                    break
            example_txt = (
                f" (e.g. {low_btw_example[0]} at betweenness {low_btw_example[1]:.5f})"
                if low_btw_example else ""
            )
            doc.add_paragraph(
                _clean_text_spaces(
                    f"Interpretation: {top['Co-Cited Author']} ranks first by combined "
                    f"citing-document count and link strength, with betweenness "
                    f"{top['Betweenness Centrality']:.5f}. Authors with high raw citation "
                    f"sums but low betweenness{example_txt} indicate widely-cited works "
                    f"that are cited within a single sub-literature rather than bridging across clusters."
                )
            )

        doc.add_heading("9. Institutional & Country Distribution", level=2)
        if img_io_cnt.getbuffer().nbytes > 0:
            doc.add_picture(img_io_cnt, width=docx.shared.Inches(5.5))
        doc.add_heading("Top Institutions", level=3)
        it = doc.add_table(rows=1, cols=3)
        it.style = "Table Grid"
        for i, h in enumerate(["Institution", "Count", "Mean Year"]):
            it.rows[0].cells[i].text = h
        for i in st.session_state.get("inst_distribution", []):
            r = it.add_row().cells
            r[0].text = str(i["Institution Name"])
            r[1].text = str(i["Publication Count"])
            r[2].text = str(i["Mean Year"])

        doc.add_heading("Top Countries (with international co-authorship)", level=3)
        ctt = doc.add_table(rows=1, cols=4)
        ctt.style = "Table Grid"
        for i, h in enumerate(["Country", "Papers", "Mean Year", "Intl Co-Auth %"]):
            ctt.rows[0].cells[i].text = h
        for c in st.session_state.get("country_distribution", []):
            r = ctt.add_row().cells
            r[0].text = str(c["Country Code"])
            r[1].text = str(c["Papers"])
            r[2].text = str(c["Mean Year"])
            r[3].text = str(c.get("Intl Co-Auth %", "N/A"))
        doc.add_paragraph(
            "Papers = number of distinct corpus papers with at least one author "
            "affiliated in that country. Intl Co-Auth % = share of those papers "
            "with authors from ≥2 countries. Counts are per paper, not per "
            "institution-affiliation."
        )

        doc.add_heading("10. Publication Source Distribution", level=2)
        if img_io_jnl.getbuffer().nbytes > 0:
            doc.add_picture(img_io_jnl, width=docx.shared.Inches(5.5))
        jt = doc.add_table(rows=1, cols=2)
        jt.style = "Table Grid"
        jt.rows[0].cells[0].text = "Journal / Source"
        jt.rows[0].cells[1].text = "Article Count"
        for j in st.session_state.get("journal_distribution", []):
            r = jt.add_row().cells
            r[0].text = str(j["Journal / Source Title"])
            r[1].text = str(j["Article Count"])

        doc.add_heading("11. Top Influential Publications & Roles", level=2)
        ptbl = doc.add_table(rows=1, cols=6)
        ptbl.style = "Table Grid"
        for i, h in enumerate(["Rank", "Title", "Year", "Cites", "Composite Score (exploratory)", "Role"]):
            ptbl.rows[0].cells[i].text = h
        for rank, (sc, w, role) in enumerate(scored_works[:15], 1):
            r = ptbl.add_row().cells
            r[0].text = str(rank)
            r[1].text = clean_title(w.get("title"))
            r[2].text = str(w.get("publication_year", "N/A"))
            r[3].text = str(w.get("cited_by_count", 0))
            r[4].text = str(sc)
            r[5].text = role

        doc.add_heading("Score Sensitivity Analysis", level=3)
        sens = st.session_state.get("score_sensitivity", {})
        if sens:
            stt = doc.add_table(rows=1, cols=3)
            stt.style = "Table Grid"
            for i, h in enumerate(["Weight scheme", "Top-15 overlap (count)", "Overlap (%)"]):
                stt.rows[0].cells[i].text = h
            for name, v in sens.items():
                r = stt.add_row().cells
                r[0].text = name
                r[1].text = str(v["top15_overlap_with_baseline"])
                r[2].text = f"{v['overlap_percent']}%"
            doc.add_paragraph(
                "The composite score is exploratory. The table above reports how many "
                "of the top-15 publications by baseline weights remain in the top-15 "
                "under alternative weight schemes. High overlap (>70%) indicates the "
                "ranking is robust; low overlap indicates the weighting choice "
                "materially changes conclusions."
            )

        doc.add_heading("Node Interconnectivity", level=2)
        nt = doc.add_table(rows=1, cols=6)
        nt.style = "Table Grid"
        for i, h in enumerate(["Rank", "Title", "Year", "Score", "Cited By (Title Excerpts)", "Cites (Title Excerpts)"]):
            nt.rows[0].cells[i].text = h
        for item in st.session_state.get("node_table_data", []):
            r = nt.add_row().cells
            r[0].text = str(item["Rank"])
            r[1].text = str(item["Title"])
            r[2].text = str(item["Year"])
            r[3].text = str(item["Score"])
            r[4].text = str(item["Cited By Nodes"])
            r[5].text = str(item["Cites Nodes"])

        doc.add_heading("Appendix: Complete Inventory of Network Nodes", level=2)
        doc.add_paragraph(f"All {len(scored_works)} publications ordered by importance score:")
        ap = doc.add_table(rows=1, cols=6)
        ap.style = "Table Grid"
        for i, h in enumerate(["Rank", "Title", "Year", "Citations", "Score", "Role"]):
            ap.rows[0].cells[i].text = h
        for rank, (sc, w, role) in enumerate(scored_works, 1):
            r = ap.add_row().cells
            r[0].text = str(rank)
            r[1].text = clean_title(w.get("title"))
            r[2].text = str(w.get("publication_year", "N/A"))
            r[3].text = str(w.get("cited_by_count", 0))
            r[4].text = str(sc)
            r[5].text = role

        doc.add_page_break()
        doc.add_heading("License & Citation", level=2)
        doc.add_heading("License", level=3)
        doc.add_paragraph(
            "NAGraph: Bibliometric & Citation Network Analysis Suite\n"
            "Copyright (c) 2026 Dr. Nibu A. George, Baselius College Kottayam (Autonomous).\n\n"
            "Licensed under CC BY 4.0: https://creativecommons.org/licenses/by/4.0/"
        )
        doc.add_heading("How to Cite NAGraph", level=3)
        doc.add_paragraph(
            "George, N. A. (2026). NAGraph: Bibliometric & Citation Network Analysis Suite "
            "(Version 1.1) [Computer software]. Baselius College Kottayam. "
            "https://github.com/nibugeorge/nagraph"
        )

        doc_io = io.BytesIO()
        doc.save(doc_io)
        doc_io.seek(0)

        st.download_button(
            label="📥 Download: NAGraph - Network Analysis Report (.docx)",
            data=doc_io,
            file_name="NAGraph - Network Analysis Report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #CBD5E1; font-size: 12px; padding: 8px;
                background-color: #1E293B; border-radius: 8px; border: 1px solid #334155;'>
        <div style='color: #F8FAFC; font-weight: bold; margin-bottom: 5px;'>🎓 How to Cite NAGraph</div>
        <div style='margin-bottom: 8px; color: #94A3B8;'>If you use this software in your research, please cite it.</div>
        <div style='font-family: monospace; background-color: #0F172A; padding: 6px;
                    border-radius: 4px; display: inline-block; border: 1px solid #475569; color: #F8FAFC;'>
            George, N. A. (2026). <i>NAGraph: Bibliometric & Citation Network Analysis Suite</i>
            (Version 1.1) [Computer software]. https://github.com/nibugeorge/nagraph
        </div>
        <div style='margin-top: 8px; font-size: 10px; color: #94A3B8;'>
            Licensed under <a href='https://creativecommons.org/licenses/by/4.0/'
            target='_blank' style='color: #60A5FA; text-decoration: none; font-weight: bold;'>CC BY 4.0</a>
            &nbsp;|&nbsp; Copyright &copy; 2026 Dr. Nibu A. George
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)