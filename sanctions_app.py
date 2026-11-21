import streamlit as st
import pandas as pd
import re
from thefuzz import fuzz
from sentence_transformers import SentenceTransformer
import chromadb

# ---------------------------------------------------------------------------
# PAGE & APP HEADER CONFIGURATION
# ---------------------------------------------------------------------------
# Set browser tab title, icon, and wide layout before rendering any UI elements
st.set_page_config(page_title="Intelligent SWIFT Sanctions Screener", page_icon="🛡️", layout="wide")

# Display the main page heading and case-study overview banner
st.title("🛡️ Automated Multi-Architecture SWIFT Sanctions Screening Engine")
st.markdown("*University Case Study: Automated Tri-Architecture Screening (Baseline Fuzzy vs. ChromaDB Vectors vs. Hybrid Pipeline)*")

# ---------------------------------------------------------------------------
# 1. LOAD & CACHE OFAC DATASETS (sdn.csv + sdn_comments.csv only)
# ---------------------------------------------------------------------------
# Cache the loaded sanctions DataFrame across Streamlit reruns and sessions
@st.cache_resource
def load_ofac_data():
    # Read the first four columns of OFAC's SDN CSV (no header row in raw file)
    sdn_df = pd.read_csv(
        'sdn.csv',
        header=None,
        usecols=[0, 1, 2, 3],
        names=['id', 'name', 'sdn_type', 'program']
    )

    try:
        # Load optional comments file containing supplementary details/aliases keyed by SDN ID
        comments_df = pd.read_csv(
            'sdn_comments.csv',
            header=None,
            usecols=[0, 1],
            names=['id', 'comments']
        )
        # Left-join comments onto SDN entries so records without comments remain intact
        sanctions_db = pd.merge(sdn_df, comments_df, on='id', how='left')
    except Exception:
        # Gracefully fall back to SDN-only data if comments file is absent or unreadable
        sanctions_db = sdn_df.copy()
        sanctions_db['comments'] = ''

    # Coerce IDs to numeric (invalid values become NaN) and normalize text fields
    sanctions_db['id'] = pd.to_numeric(sanctions_db['id'], errors='coerce')
    sanctions_db['name_clean'] = sanctions_db['name'].fillna('').astype(str).str.strip()
    sanctions_db['type_clean'] = sanctions_db['sdn_type'].fillna('').astype(str).str.strip()
    sanctions_db['program_clean'] = sanctions_db['program'].fillna('Unknown').astype(str).str.strip()
    sanctions_db['comments_clean'] = sanctions_db['comments'].fillna('').astype(str).str.strip()

    # Combine cleaned fields into a single searchable text string for embedding and display
    sanctions_db['search_text'] = (
        sanctions_db['name_clean'] +
        " | Type: " + sanctions_db['type_clean'].replace('', 'Entity/Individual') +
        " | Program: " + sanctions_db['program_clean'] +
        " | Comments: " + sanctions_db['comments_clean'].replace('', 'None')
    )

    # Select relevant columns, drop exact duplicate rows, and reset the DataFrame index
    corpus = sanctions_db[
        ['id', 'name_clean', 'type_clean', 'program_clean', 'comments_clean', 'search_text']
    ].drop_duplicates().reset_index(drop=True)
    return corpus

# Show a status spinner while loading and cleaning the OFAC sanctions corpus
with st.spinner("Loading and indexing OFAC SDN & comments records..."):
    corpus = load_ofac_data()

# Cache the embedding model and ChromaDB collection so indexing runs once per app process
@st.cache_resource
def init_vector_db():
    # Instantiate the MiniLM sentence-transformer model for dense semantic embeddings
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Create an in-memory ChromaDB client and get or create the target collection
    client = chromadb.Client()
    col = client.get_or_create_collection(name="ofac_swift_sdn_comments_v1")
    
    # Index the first 3,000 corpus records with unique IDs combining SDN ID and row index
    sample = corpus.head(3000).copy()
    unique_ids = [f"{row['id']}_{i}" for i, row in sample.iterrows()]
    col.add(documents=sample['search_text'].tolist(), ids=unique_ids)
    return embedder, col

# Initialize the embedder and vector collection for semantic similarity queries
embedder, collection = init_vector_db()

# ---------------------------------------------------------------------------
# 2. INTERACTIVE CUSTOMER PAYMENT PORTAL (SWIFT MT103 Generator)
# ---------------------------------------------------------------------------
# Build the left sidebar form for constructing or pre-filling a SWIFT MT103 message
st.sidebar.header("💳 Customer Payment Portal (SWIFT MT103)")
st.sidebar.markdown("Create a custom payment or load a preset scenario:")

# Let the user choose between manual entry, sanction test cases, or a clean payment
preset = st.sidebar.selectbox(
    "Load Scenario Preset (Optional):",
    [
        "Custom Input (Type Your Own)",
        "Test Sanction: Aerocaribbean Airlines",
        "Test Sanction: Banco Nacional de Cuba",
        "Test Clean: Global Tech Solutions"
    ]
)

# Set default payment fields according to the chosen preset scenario
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
    def_sender = "ISLAMBOULI, Mohammad Shawqi\nCHICAGO, CUBA"
    def_beneficiary = "GLOBAL PARTNERS LTD\nZURICH, SWITZERLAND"
    def_amount = "25000.00"
    def_currency = "USD"
    def_remittance = "CONSULTING SERVICES Q4"

# Render sidebar input controls for ordering customer, beneficiary, amount, banks, and remittance
sender_name = st.sidebar.text_area("Ordering Customer (Field 50K):", value=def_sender, height=75)
beneficiary_name = st.sidebar.text_area("Beneficiary Customer (Field 59):", value=def_beneficiary, height=75)

col_curr, col_amt = st.sidebar.columns([1, 2])
currency = col_curr.selectbox("Currency", ["USD", "EUR", "GBP", "CHF", "CAD"], index=0 if def_currency=="USD" else 1)
amount = col_amt.text_input("Amount (Field 32A)", value=def_amount)

intermediary = st.sidebar.text_input("Intermediary Bank BIC (Field 56A)", value="BOFAUS3NXXX")
remittance = st.sidebar.text_input("Remittance Info (Field 70)", value=def_remittance)

# Format the fields into a raw SWIFT MT103 wire message string (converting newlines to ' | '
# and replacing decimal dots with commas in Field 32A)
swift_text = f""":20:REF2026-PAYMENT-01
:23B:CRED
:32A:241115{currency}{amount.replace('.', ',')}
:50K:{sender_name.replace(chr(10), ' | ')}
:53A:CHASUS33XXX
:56A:{intermediary}
:57A:CITIUS33XXX
:59:{beneficiary_name.replace(chr(10), ' | ')}
:70:{remittance}"""

# Display a live code preview of the assembled SWIFT message in the sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Generated SWIFT Preview:")
st.sidebar.code(swift_text, language="text")

# ---------------------------------------------------------------------------
# 3. LIST-GROUNDED MULTI-ARCHITECTURE SCREENING (sdn.csv + sdn_comments.csv)
# ---------------------------------------------------------------------------
def parse_swift(text):
    # Parse `:TAG:VALUE` segments up to the next `:TAG:` line or end of message
    fields = {}
    pattern = r':(\w+):(.*?)(?=\n:\w+:|\Z)'
    matches = re.findall(pattern, text, re.DOTALL)
    for tag, val in matches:
        # Collapse line breaks inside a field into ' | ' delimiters
        fields[tag] = val.strip().replace('\n', ' | ')
    return fields

# Extract tag-to-value mapping from the current SWIFT message
parsed_fields = parse_swift(swift_text)

# Split target fields (50K ordering customer, 59 beneficiary, 70 remittance) by '|' or ','
# into candidate entity/phrase strings of at least 3 characters
candidates = []
for tag in ['50K', '59', '70']:
    if tag in parsed_fields:
        raw_val = parsed_fields[tag]
        parts = [p.strip() for p in re.split(r'\||,', raw_val) if len(p.strip()) >= 3]
        for p in parts:
            candidates.append((tag, p))

# Deduplicate candidates per field tag using case-insensitive keys while keeping first-seen text
seen_candidates = set()
entities_to_screen = []
for tag, cand in candidates:
    key = (tag, cand.upper())
    if key not in seen_candidates:
        seen_candidates.add(key)
        entities_to_screen.append((tag, cand))

screening_results = []
# Limit fuzzy scanning to the first 2,000 corpus entries for faster interactive response
search_subset = corpus.head(2000)

# Evaluate each extracted candidate across fuzzy, vector, and hybrid screening rules
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
                # token_set_ratio compares unique word sets, handling reordered or partial tokens
                score = fuzz.token_set_ratio(ent_lower, tf.lower())
                if score > best_fuzzy_score:
                    best_fuzzy_score = score
                    best_fuzzy_match = row['search_text']
                    best_fuzzy_field = field_label

    # Architecture 2: Pure Vector Search (ChromaDB)
    # Retrieve the top-1 nearest vector match and distance for the candidate string
    vector_res = collection.query(query_texts=[entity], n_results=1)
    top_vec_doc = vector_res['documents'][0][0] if vector_res['documents'] and vector_res['documents'][0] else "No match"
    vector_dist = vector_res['distances'][0][0] if vector_res['distances'] and vector_res['distances'][0] else 99.0
    # Treat vector result as a match only when distance is strictly less than 0.65
    vector_matched_doc = top_vec_doc if vector_dist < 0.65 else "No vector match within threshold (< 0.65)"

    # Architecture 3: Hybrid Adjudication strictly from uploaded list match strength
    # Flag the candidate if either fuzzy score >= 85 or vector distance < 0.65
    fuzzy_hit = best_fuzzy_score >= 85
    vector_hit = vector_dist < 0.65
    is_flagged = fuzzy_hit or vector_hit

    # Format the primary matched OFAC record summary for display
    if fuzzy_hit:
        exact_matched_record = f"[{best_fuzzy_field} Match] {best_fuzzy_match}"
    elif vector_hit:
        exact_matched_record = f"[Vector Match] {top_vec_doc}"
    else:
        exact_matched_record = "No matching OFAC record"

    # Record all scoring metrics and match details for this candidate
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
# Split main workspace into two columns: message view on the left, hit review on the right
col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.subheader("📄 SWIFT MT103 Payment Breakdown & Highlighted Hits")
    
    # Highlight every flagged entity text inside the SWIFT message using a red <mark> span
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
    
    # Render the styled monospace SWIFT message container with highlighted hits
    st.markdown(f"""
    <div style="background-color: #0e1117; border: 1px solid #30363d; padding: 15px; border-radius: 8px; font-family: monospace; white-space: pre-wrap;">
{highlighted_display}
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.subheader("🎯 Compliance Hits & Interactive Navigator")
    
    # Collect only candidates that triggered a sanction flag
    flagged_hits = [r for r in screening_results if r["flagged"]]
    
    if not flagged_hits:
        # Display green all-clear message when no candidates crossed hit thresholds
        st.success("🟢 **NO SANCTION HITS DETECTED**\n\nNo matching entities, individuals, or entries found in `sdn.csv` or `sdn_comments.csv`.")
    else:
        # Display red alert header with the count of automatically flagged hits
        st.error(f"🔴 **{len(flagged_hits)} SANCTION ALERT(S) FLAGGED AUTOMATICALLY**")
        
        # Initialize session state dictionary to persist manual false-positive dismissals
        if 'discounted' not in st.session_state:
            st.session_state.discounted = {}

        # Create a tab for each flagged hit so reviewers can inspect them one by one
        hit_tabs = st.tabs([f"Hit #{i+1} ({h['entity']})" for i, h in enumerate(flagged_hits)])
        
        for i, (tab, hit) in enumerate(zip(hit_tabs, flagged_hits)):
            with tab:
                # Show the extracted candidate string and which SWIFT field it came from
                st.markdown(f"**Extracted Payment Entity (`Field {hit['tag']}`):**")
                st.code(hit['entity'])
                
                # Show fuzzy similarity score and vector distance alongside their thresholds
                st.markdown("#### 🔬 Tri-Architecture Scoring:")
                col_a, col_b = st.columns(2)
                col_a.metric("Baseline Fuzzy Score", f"{hit['fuzzy_score']}%", "Threshold: 85%")
                col_b.metric("Vector Distance", f"{hit['vector_distance']:.4f}", "Threshold: < 0.65")
                
                # Display the matched OFAC database record string
                st.markdown("#### ⚖️ Exact List Matching Result:")
                st.markdown(f"**Matched OFAC Database Record:**\n> `{hit['exact_matched_record']}`")
                
                # Key each hit by field tag and entity text to track manual discount status
                hit_key = f"{hit['tag']}_{hit['entity']}"
                is_discounted = st.session_state.discounted.get(hit_key, False)
                
                # Provide buttons to discount a false positive or restore a previously discounted hit
                if is_discounted:
                    st.warning("⚠️ **Hit Manually Discounted / Dismissed as False Positive by Compliance Officer.**")
                    if st.button(f"Restore Hit #{i+1}", key=f"restore_{i}"):
                        st.session_state.discounted[hit_key] = False
                        st.rerun()
                else:
                    if st.button(f"Discount / Dismiss Hit #{i+1} (False Positive)", key=f"discount_{i}", type="secondary"):
                        st.session_state.discounted[hit_key] = True
                        st.rerun()