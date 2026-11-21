import streamlit as st
import pandas as pd
import re
from thefuzz import fuzz
from sentence_transformers import SentenceTransformer
import chromadb

# ---------------------------------------------------------------------------
# PAGE & APP HEADER CONFIGURATION
# ---------------------------------------------------------------------------
# Configure Streamlit browser tab metadata and use a wide layout for side-by-side panels
st.set_page_config(page_title="Intelligent SWIFT Sanctions Screener", page_icon="🛡️", layout="wide")

# Render main application title and subtitle describing the three compared screening methods
st.title("🛡️ Automated Multi-Architecture SWIFT Sanctions Screening Engine")
st.markdown("*University Case Study: Automated Tri-Architecture Screening (Baseline Fuzzy vs. ChromaDB Vectors vs. Hybrid Pipeline)*")

# ---------------------------------------------------------------------------
# 1. LOAD & CACHE OFAC DATASETS (sdn.csv + sdn_comments.csv only)
# ---------------------------------------------------------------------------
# Cache the resulting sanctions corpus across reruns and user sessions to avoid re-reading CSVs
@st.cache_resource
def load_ofac_data():
    # Load the primary OFAC Specially Designated Nationals (SDN) list without headers; read all fields as strings
    sdn_df = pd.read_csv(
        'sdn.csv',
        header=None,
        usecols=[0, 1, 2, 3],
        names=['id', 'name', 'sdn_type', 'program'],
        dtype=str,
        keep_default_na=False
    )

    # Normalize the record ID by stripping surrounding whitespace for reliable joins
    sdn_df['id_key'] = sdn_df['id'].str.strip()

    try:
        # Attempt to load optional SDN comments containing aliases, identifiers, or notes
        comments_df = pd.read_csv(
            'sdn_comments.csv',
            header=None,
            usecols=[0, 1],
            names=['id', 'comments'],
            dtype=str,
            keep_default_na=False
        )
        comments_df['id_key'] = comments_df['id'].str.strip()

        # Group multiple comment rows per SDN ID into a single deduplicated, sorted semicolon-delimited string
        comments_agg = (
            comments_df.groupby('id_key')['comments']
            .apply(lambda s: " ; ".join(sorted({x.strip() for x in s if x.strip()})))
            .reset_index()
        )

        # Left-join aggregated comments onto the main SDN table so records without comments are preserved
        sanctions_db = pd.merge(
            sdn_df,
            comments_agg[['id_key', 'comments']],
            on='id_key',
            how='left'
        )
    except Exception:
        # Fall back cleanly if sdn_comments.csv is missing or unreadable
        sanctions_db = sdn_df.copy()
        sanctions_db['comments'] = ''

    # Clean whitespace and supply sensible defaults for missing SDN fields
    sanctions_db['name_clean'] = sanctions_db['name'].fillna('').str.strip()
    sanctions_db['type_clean'] = sanctions_db['sdn_type'].fillna('').str.strip()
    sanctions_db['program_clean'] = sanctions_db['program'].replace('', 'Unknown').fillna('Unknown').str.strip()
    sanctions_db['comments_clean'] = sanctions_db['comments'].fillna('').str.strip()

    # Build a unified text representation combining key metadata for vector embedding and display
    sanctions_db['search_text'] = (
        sanctions_db['name_clean'] +
        " | Type: " + sanctions_db['type_clean'].replace('', 'Entity/Individual') +
        " | Program: " + sanctions_db['program_clean'] +
        " | Comments: " + sanctions_db['comments_clean'].replace('', 'None')
    )

    # Keep only the columns needed downstream and remove exact duplicate rows
    corpus = sanctions_db[
        ['id_key', 'name_clean', 'type_clean', 'program_clean', 'comments_clean', 'search_text']
    ].drop_duplicates().reset_index(drop=True)
    return corpus


# Display a progress spinner while loading and preparing the OFAC dataset
with st.spinner("Loading and indexing OFAC SDN & comments records..."):
    corpus = load_ofac_data()


# Cache the SentenceTransformer model and in-memory ChromaDB vector collection across reruns
@st.cache_resource
def init_vector_db(corpus_df):
    # Load lightweight semantic embedding model for dense vector representations
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Initialize an ephemeral ChromaDB client and create or retrieve the target collection
    client = chromadb.Client()
    col = client.get_or_create_collection(name="ofac_swift_sdn_comments_v2")

    # Populate the collection only when empty (first initialization) using up to the first 3,000 records
    if col.count() == 0 and len(corpus_df) > 0:
        sample = corpus_df.head(3000).copy()
        unique_ids = [f"{row['id_key']}_{i}" for i, row in sample.iterrows()]
        docs = sample['search_text'].tolist()

        # Insert documents in batches of 500 to keep indexing smooth and bounded
        batch_size = 500
        for start in range(0, len(docs), batch_size):
            col.add(
                documents=docs[start:start + batch_size],
                ids=unique_ids[start:start + batch_size]
            )
    return embedder, col


# Initialize the embedding model and ChromaDB collection once
embedder, collection = init_vector_db(corpus)

# ---------------------------------------------------------------------------
# 2. INTERACTIVE CUSTOMER PAYMENT PORTAL (SWIFT MT103 Generator)
# ---------------------------------------------------------------------------
# Build sidebar controls for composing or presetting a SWIFT MT103 wire transfer message
st.sidebar.header("💳 Customer Payment Portal (SWIFT MT103)")
st.sidebar.markdown("Create a custom payment or load a preset scenario:")

# Let the user pick between freeform input, known sanctioned entities, or a clean payment scenario
preset = st.sidebar.selectbox(
    "Load Scenario Preset (Optional):",
    [
        "Custom Input (Type Your Own)",
        "Test Sanction: Aerocaribbean Airlines",
        "Test Sanction: Banco Nacional de Cuba",
        "Test Clean: Global Tech Solutions"
    ]
)

# Populate default field values based on the selected preset scenario
if "Aerocaribbean" in preset:
    def_sender = "GLOBAL EXPORT CORP\nNEW YORK, USA"
    def_beneficiary = "AEROCARIBBEAN AIRLINES\nHAVANA, CUBA"
    def_amount = "150000.00"
    def_currency = "USD"
    def_remittance = "PAYMENT FOR AIRCRAFT SPARE PARTS - REF INV 8891"
elif "Banco" in preset:
    def_sender = "EUROPEAN TRADING GMBH\nFRANKFURT, GERMANY"
    def_beneficiary = "BANCO NACIONAL DE CUBA\nHAVANA, CUBA"
    def_amount = "50000.00"
    def_currency = "EUR"
    def_remittance = "SETTLEMENT OF TREASURY CLEARING BALANCES"
elif "Clean" in preset:
    def_sender = "ACME TECHNOLOGIES LLC\nSAN FRANCISCO, USA"
    def_beneficiary = "SILICON VALLEY SOFTWARE SOLUTIONS\nLONDON, UK"
    def_amount = "12500.00"
    def_currency = "USD"
    def_remittance = "MONTHLY CLOUD INFRASTRUCTURE FEES"
else:
    def_sender = "MOHAMED AHMED\nCHICAGO, USA"
    def_beneficiary = "GLOBAL PARTNERS LTD\nZURICH, SWITZERLAND"
    def_amount = "25000.00"
    def_currency = "USD"
    def_remittance = "CONSULTING SERVICES Q4"

# Sidebar input widgets corresponding to key SWIFT MT103 fields
sender_name = st.sidebar.text_area("Ordering Customer (Field 50K):", value=def_sender, height=75)
beneficiary_name = st.sidebar.text_area("Beneficiary Customer (Field 59):", value=def_beneficiary, height=75)

# Side-by-side currency selector and payment amount input
col_curr, col_amt = st.sidebar.columns([1, 2])
currency = col_curr.selectbox("Currency", ["USD", "EUR", "GBP", "CHF", "CAD"], index=0 if def_currency == "USD" else 1)
amount = col_amt.text_input("Amount (Field 32A)", value=def_amount)

intermediary = st.sidebar.text_input("Intermediary Bank BIC (Field 56A)", value="BOFAUS3NXXX")
remittance = st.sidebar.text_input("Remittance Info (Field 70)", value=def_remittance)

# Assemble the raw SWIFT MT103 message string (replacing newlines in multi-line fields with ' | '
# and converting decimal points to commas in Field 32A per SWIFT conventions)
swift_text = f""":20:REF2026-PAYMENT-01
:23B:CRED
:32A:241115{currency}{amount.replace('.', ',')}
:50K:{sender_name.replace(chr(10), ' | ')}
:53A:CHASUS33XXX
:56A:{intermediary}
:57A:CITIUS33XXX
:59:{beneficiary_name.replace(chr(10), ' | ')}
:70:{remittance}"""

# Show a live preview of the generated SWIFT message in the sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Generated SWIFT Preview:")
st.sidebar.code(swift_text, language="text")

# ---------------------------------------------------------------------------
# 3. LIST-GROUNDED MULTI-ARCHITECTURE SCREENING (sdn.csv + sdn_comments.csv)
# ---------------------------------------------------------------------------
def parse_swift(text):
    # Extract `:TAG:VALUE` blocks up to the next `:TAG:` line or end of text
    fields = {}
    pattern = r':(\w+):(.*?)(?=\n:\w+:|\Z)'
    matches = re.findall(pattern, text, re.DOTALL)
    for tag, val in matches:
        # Normalize internal line breaks into ' | ' separators
        fields[tag] = val.strip().replace('\n', ' | ')
    return fields


# Parse all SWIFT tags from the generated message
parsed_fields = parse_swift(swift_text)

# Split high-risk party and remittance fields (50K, 59, 70) on '|' or ',' into candidate strings
candidates = []
for tag in ['50K', '59', '70']:
    if tag in parsed_fields:
        raw_val = parsed_fields[tag]
        parts = [p.strip() for p in re.split(r'\||,', raw_val) if len(p.strip()) >= 3]
        for p in parts:
            candidates.append((tag, p))

# Deduplicate candidates per field tag (case-insensitively) while preserving original casing and order
seen_candidates = set()
entities_to_screen = []
for tag, cand in candidates:
    key = (tag, cand.upper())
    if key not in seen_candidates:
        seen_candidates.add(key)
        entities_to_screen.append((tag, cand))

screening_results = []
# Restrict screening comparisons to the first 3,000 corpus entries to align with the indexed sample
search_subset = corpus.head(3000)

# Screen each extracted candidate string across all three matching architectures
for tag, entity in entities_to_screen:
    ent_lower = entity.lower()

    # Architecture 1: Baseline Fuzzy Match against SDN Name, Program, and Comments
    best_fuzzy_score = 0
    best_fuzzy_match = "No list match"
    best_fuzzy_field = ""

    for _, row in search_subset.iterrows():
        field_checks = [
            ("SDN Name", row['name_clean']),
            ("Program", row['program_clean']),
            ("Comments", row['comments_clean']),
        ]
        for field_label, tf in field_checks:
            if tf:
                # token_set_ratio tolerates word-order differences and extra/subset tokens
                score = fuzz.token_set_ratio(ent_lower, tf.lower())
                if score > best_fuzzy_score:
                    best_fuzzy_score = score
                    best_fuzzy_match = row['search_text']
                    best_fuzzy_field = field_label

    # Architecture 2: Pure Vector Search (ChromaDB)
    # Query ChromaDB for the single nearest neighbor document to the candidate string
    vector_res = collection.query(query_texts=[entity], n_results=1)
    top_vec_doc = vector_res['documents'][0][0] if vector_res['documents'] and vector_res['documents'][0] else "No match"
    vector_dist = vector_res['distances'][0][0] if vector_res['distances'] and vector_res['distances'][0] else 99.0
    # Accept a vector match only if its distance falls below the 0.65 similarity threshold
    vector_matched_doc = top_vec_doc if vector_dist < 0.65 else "No vector match within threshold (< 0.65)"

    # Architecture 3: Hybrid Adjudication strictly from uploaded list match strength
    # Flag if either fuzzy similarity >= 85% or vector distance < 0.65
    fuzzy_hit = best_fuzzy_score >= 85
    vector_hit = vector_dist < 0.65
    is_flagged = fuzzy_hit or vector_hit

    # Prefer explicit fuzzy match details when available, falling back to vector match if only that hit
    if fuzzy_hit:
        exact_matched_record = f"[{best_fuzzy_field} Match] {best_fuzzy_match}"
    elif vector_hit:
        exact_matched_record = f"[Vector Match] {top_vec_doc}"
    else:
        exact_matched_record = "No matching OFAC record"

    # Store per-candidate scores, matches, and adjudication outcome
    screening_results.append({
        "tag": tag,
        "entity": entity,
        "fuzzy_score": best_fuzzy_score,
        "fuzzy_match": best_fuzzy_match,
        "vector_distance": vector_dist,
        "vector_match": vector_matched_doc,
        "exact_matched_record": exact_matched_record,
        "flagged": is_flagged
    })

# ---------------------------------------------------------------------------
# 4. UI LAYOUT: HIGHLIGHTED MESSAGE & INTERACTIVE HITS NAVIGATOR
# ---------------------------------------------------------------------------
# Create two main columns: left for the SWIFT message with highlights, right for hit review
col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.subheader("📄 SWIFT MT103 Payment Breakdown & Highlighted Hits")

    # Wrap every flagged candidate string in a red HTML <mark> tag (case-insensitive)
    highlighted_display = swift_text
    for res in screening_results:
        if res["flagged"]:
            pattern_esc = re.escape(res["entity"])
            highlighted_display = re.sub(
                pattern_esc,
                f"<mark style='background-color: #ff4b4b; color: white; padding: 2px 6px; border-radius: 4px;'><b>{res['entity']}</b></mark>",
                highlighted_display,
                flags=re.IGNORECASE
            )

    # Render the monospace SWIFT message box with inline hit highlights
    st.markdown(f"""
    <div style="background-color: #0e1117; border: 1px solid #30363d; padding: 15px; border-radius: 8px; font-family: monospace; white-space: pre-wrap;">
{highlighted_display}
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.subheader("🎯 Compliance Hits & Interactive Navigator")

    # Filter down to candidates that triggered a sanction flag
    flagged_hits = [r for r in screening_results if r["flagged"]]

    if not flagged_hits:
        # Show clear banner when no candidate met the fuzzy or vector hit thresholds
        st.success("🟢 **NO SANCTION HITS DETECTED**\n\nNo matching entities, individuals, or entries found in `sdn.csv` or `sdn_comments.csv`.")
    else:
        # Alert banner summarizing how many automatic sanction hits were found
        st.error(f"🔴 **{len(flagged_hits)} SANCTION ALERT(S) FLAGGED AUTOMATICALLY**")

        # Track compliance officers' manual false-positive dismissals across reruns in session state
        if 'discounted' not in st.session_state:
            st.session_state.discounted = {}

        # Build one tab per flagged hit for focused inspection
        hit_tabs = st.tabs([f"Hit #{i+1} ({h['entity']})" for i, h in enumerate(flagged_hits)])

        for i, (tab, hit) in enumerate(zip(hit_tabs, flagged_hits)):
            with tab:
                # Display the extracted text and originating SWIFT field tag
                st.markdown(f"**Extracted Payment Entity (`Field {hit['tag']}`):**")
                st.code(hit['entity'])

                # Display scores and thresholds from the fuzzy and vector architectures
                st.markdown("#### 🔬 Tri-Architecture Scoring:")
                col_a, col_b = st.columns(2)
                col_a.metric("Baseline Fuzzy Score", f"{hit['fuzzy_score']}%", "Threshold: 85%")
                col_b.metric("Vector Distance", f"{hit['vector_distance']:.4f}", "Threshold: < 0.65")

                # Show the specific OFAC record or match string that triggered the result
                st.markdown("#### ⚖️ Exact List Matching Result:")
                st.markdown(f"**Matched OFAC Database Record:**\n> `{hit['exact_matched_record']}`")

                # Unique key per hit so dismissal state persists per field + entity pair
                hit_key = f"{hit['tag']}_{hit['entity']}"
                is_discounted = st.session_state.discounted.get(hit_key, False)

                # Toggle between dismissing a hit as a false positive and restoring it
                if is_discounted:
                    st.warning("⚠️ **Hit Manually Discounted / Dismissed as False Positive by Compliance Officer.**")
                    if st.button(f"Restore Hit #{i+1}", key=f"restore_{i}"):
                        st.session_state.discounted[hit_key] = False
                        st.rerun()
                else:
                    if st.button(f"Discount / Dismiss Hit #{i+1} (False Positive)", key=f"discount_{i}", type="secondary"):
                        st.session_state.discounted[hit_key] = True
                        st.rerun()