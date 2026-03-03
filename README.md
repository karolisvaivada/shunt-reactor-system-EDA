# Shunt Reactor System -- EDA & Monitoring

## Project Overview

This project analyzes operational data from a high-voltage shunt
reactor. The goal is to evaluate reactor behavior, detect abnormal
events, assess data quality, and verify compliance with design
specifications.

The analysis includes:

-   Voltage, current and reactive power monitoring
-   Timestamp gap detection
-   Full data loss (zero block) detection
-   Reactor OFF event detection
-   Design limit violation analysis
-   IQR-based outlier detection
-   Voltage and current imbalance calculation
-   Interactive cloud visualisation

------------------------------------------------------------------------

## Key Files

-   `results.ipynb` -- Full presentation and analysis notebook\
-   `src/functions.py` -- Core processing and visualisation functions\
-   `src/app.py` -- Streamlit cloud application\
-   `requirements.txt` -- Project dependencies

------------------------------------------------------------------------

## How to Use This Repository (GitHub Guide)

1. Open the repository on GitHub.
2. Click on **`results.ipynb`** to view the full analysis and presentation.
3. Scroll through the notebook to review:
   - Data overview
   - Event detection results
   - Reactor behavior analysis
   - Conclusions

4. For interactive visualisations:

   Open the Streamlit Cloud dashboard:

   🔗 https://karolisvaivada-shunt-reactor-system-eda-srcapp-rwisyt.streamlit.app/

The cloud dashboard allows you to:
- Zoom into specific time periods
- Inspect reactor OFF events
- View full data loss blocks
- Analyze limit violations
- Review imbalance metrics interactively

No local setup is required to view results.

------------------------------------------------------------------------

## How to Run Locally

### 1. Clone repository

``` bash
git clone <your-repository-url>
cd <repository-folder>
```

### 2. Create virtual environment

``` bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate    # Windows
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Run analysis notebook

Open:

    results.ipynb

Run all cells to reproduce full analysis and event report.

### 5. Run Streamlit app locally

``` bash
streamlit run src/app.py
```

------------------------------------------------------------------------

## Methodology Summary

-   Timestamp gaps detected using dominant sampling interval.
-   Full data loss defined as rows where all numeric signals equal zero.
-   Reactor OFF defined as all phase currents = 0 while voltage present.
-   Design limits:
    -   Voltage: 27--33 kV
    -   Current: 112.6--137.6 A
-   IQR method used to detect abnormal values outside statistical range.
-   Imbalance calculated using max--min over mean (%).

------------------------------------------------------------------------

## Assumptions

-   Nominal system voltage: 30 kV
-   Nominal phase current: 125.1 A
-   Extreme spikes (e.g., 1300 A, 300 kV) treated as non-physical
    scaling/telemetry errors.

------------------------------------------------------------------------

## Author

Karolis Vaivada\
Electrical Engineering & Data Science