# Account Takeover Detection

ML pipeline for detecting account takeover (ATO) events in login data.

## Project Structure

```
src/data/      # Loading, quality gate, cleaning
src/features/  # Feature engineering
src/models/    # Training and prediction
app/           # FastAPI + Streamlit dashboard
tests/         # Unit tests
notebooks/     # EDA
```

## Exploratory Data Analysis

**Dataset:** 31.27M login attempts × 16 features. Synthesized from a Norwegian SSO service (Feb 2020 – Feb 2021). Features span IP geolocation (Country, Region, City, ASN), device fingerprint (OS, Browser, Device Type), and session metadata (Timestamp, RTT, Login Successful). Target: `Is Account Takeover` (bool).

**Key findings:**

- **Extreme class imbalance — 221,000:1 ratio.** Only 141 ATO events in 31.27M logins (0.00045%). Accuracy is meaningless; use precision-recall AUC. Models will need class weighting or SMOTE.
- **`Round-Trip Time [ms]` is 95.9% null.** RTT was only recorded for ~4% of sessions. Treat as a binary presence indicator rather than a continuous feature, or drop entirely.
- **`Is Attack IP` is a near-perfect ATO predictor.** Every one of the 141 ATO rows has `Is Attack IP = True`. This flag is a post-hoc incident-response label — exclude it from training features or the model learns a data leak, not behavior.
- **User IDs are hashed pseudonyms** spanning the full int64 range with no ordinal meaning. Use frequency or target encoding, not raw numeric values.
- **Geographic features are nearly complete** (<0.2% null) and are the strongest candidate features for behavioral anomaly detection (e.g., login from an unusual country for a given user).
