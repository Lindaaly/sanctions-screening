# Intelligent Sanctions and PEP Screening

An educational Streamlit application for screening SWIFT MT103 payment messages against OFAC sanctions data. The app compares three approaches:

- Baseline fuzzy-name matching
- ChromaDB vector similarity
- A hybrid fuzzy and semantic pipeline

## Project files

- `sanctions_app.py` - primary Streamlit application
- `sanctions_screening.py` - screening module and application code
- `sdn.csv` - OFAC Specially Designated Nationals dataset
- `sdn_comments.csv` - supplementary OFAC SDN comments

## Requirements

Install the Python dependencies:

```powershell
pip install streamlit pandas thefuzz sentence-transformers chromadb
```

## Run the application

From the project directory:

```powershell
streamlit run sanctions_app.py
```

The application loads the OFAC CSV datasets from the project directory and provides interactive SWIFT MT103 screening and match review.
