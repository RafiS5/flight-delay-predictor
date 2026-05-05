"""
Flight Delay Predictor — Streamlit UI

Run with: `streamlit run streamlit_app.py` from the project root.
"""
from __future__ import annotations

import calendar

import streamlit as st

import inference

st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Light styling — small CSS block to give the page a polished feel without
# pulling in any extra dependencies.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 760px; }
      h1 { font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.2rem; }
      .subtitle { color: #6b7280; font-size: 0.95rem; margin-bottom: 1.8rem; }
      .result-card {
        border: 1px solid rgba(120, 120, 120, 0.18);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-top: 1.2rem;
        background: rgba(250, 250, 252, 0.6);
      }
      .result-verdict { font-size: 1.55rem; font-weight: 700; margin: 0.2rem 0 0.4rem; }
      .result-prob { font-size: 1rem; color: #4b5563; }
      .input-summary { font-size: 0.82rem; color: #6b7280; margin-top: 0.8rem; }
      .stButton button { font-weight: 600; padding: 0.55rem 1.2rem; }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Lookups & airline display names
# ---------------------------------------------------------------------------
AIRLINE_NAMES = {
    "AA": "American Airlines", "DL": "Delta Air Lines", "UA": "United Airlines",
    "WN": "Southwest Airlines", "B6": "JetBlue Airways", "AS": "Alaska Airlines",
    "NK": "Spirit Airlines", "F9": "Frontier Airlines", "HA": "Hawaiian Airlines",
    "VX": "Virgin America", "G4": "Allegiant Air", "9E": "Endeavor Air",
    "OH": "PSA Airlines", "OO": "SkyWest Airlines", "MQ": "Envoy Air",
    "EV": "ExpressJet", "YX": "Republic Airways", "YV": "Mesa Airlines",
    "ZW": "Air Wisconsin", "CP": "Compass Airlines", "C5": "CommuteAir",
    "AX": "Trans States", "EM": "Empire Airlines", "KS": "PenAir",
    "9K": "Cape Air", "G7": "GoJet Airlines", "PT": "Piedmont Airlines",
    "QX": "Horizon Air",
}


@st.cache_data(show_spinner=False)
def get_form_options():
    lookups = inference.load_lookups()
    airports = inference.load_airports()

    airline_codes = sorted(lookups["airline"]["AIRLINE"].tolist())
    airline_options = [
        (code, f"{AIRLINE_NAMES.get(code, code)} ({code})") for code in airline_codes
    ]
    # Default to a major carrier if present
    default_airline_idx = next(
        (i for i, (c, _) in enumerate(airline_options) if c == "AA"), 0
    )

    # Restrict the airport dropdowns to IATA codes the model has actually seen
    # in training, so the lookups always resolve and the model is never asked
    # to predict for an unfamiliar airport category.
    origins = set(lookups["origin"]["ORIGIN"].tolist())
    dests = set(lookups["dest"]["DEST"].tolist())
    valid_iatas = origins | dests
    airports_in_data = airports[airports["IATA_CODE"].isin(valid_iatas)].copy()
    airports_in_data["label"] = (
        airports_in_data["IATA_CODE"] + " — " + airports_in_data["AIRPORT"]
    )
    airports_in_data = airports_in_data.sort_values("IATA_CODE").reset_index(drop=True)

    weather_labels = ["Auto (typical for season)"] + [
        k for k in lookups["weather"]["presets"].keys() if not k.startswith("Auto")
    ]
    return {
        "airlines": airline_options,
        "default_airline_idx": default_airline_idx,
        "airports": airports_in_data,
        "weather_labels": weather_labels,
    }


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# ✈️ Flight Delay Predictor")
st.markdown(
    "<div class='subtitle'>FT-Transformer model · F1 0.49 · ROC-AUC 0.86</div>",
    unsafe_allow_html=True,
)

st.info(
    "**Historical model demo** — the FT-Transformer was trained on Jan–Oct 2018 BTS flight "
    "data with 2019 held out for testing. Predictions reflect typical 2018–2019 operating "
    "patterns for the chosen airline / route / time, not live or forward-looking forecasts."
)

opts = get_form_options()
airports_df = opts["airports"]

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
with st.form("flight_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        origin_label = st.selectbox(
            "Origin airport",
            options=airports_df["label"].tolist(),
            index=int(airports_df.index[airports_df["IATA_CODE"] == "JFK"][0])
            if (airports_df["IATA_CODE"] == "JFK").any() else 0,
        )
    with col2:
        dest_label = st.selectbox(
            "Destination airport",
            options=airports_df["label"].tolist(),
            index=int(airports_df.index[airports_df["IATA_CODE"] == "LAX"][0])
            if (airports_df["IATA_CODE"] == "LAX").any() else 1,
        )

    col3, col4 = st.columns(2)
    with col3:
        airline_idx = st.selectbox(
            "Airline",
            options=list(range(len(opts["airlines"]))),
            index=opts["default_airline_idx"],
            format_func=lambda i: opts["airlines"][i][1],
        )
    with col4:
        hour = st.slider("Departure hour", 0, 23, 8, format="%d:00")

    col5, col6 = st.columns(2)
    with col5:
        month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            index=0,
            format_func=lambda m: calendar.month_name[m],
            help="The model consumes month rather than a calendar date — the year is not a model feature.",
        )
    with col6:
        DOW_LABELS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_of_week = st.selectbox(
            "Day of week",
            options=list(range(1, 8)),
            index=0,
            format_func=lambda d: DOW_LABELS[d - 1],
        )

    col7, _ = st.columns([3, 2])
    with col7:
        weather_label = st.selectbox(
            "Weather conditions",
            options=opts["weather_labels"],
            index=0,
            help="'Auto' uses typical conditions for the season. Other presets use realistic values from training data.",
        )

    submitted = st.form_submit_button("Predict delay", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if submitted:
    origin_iata = origin_label.split(" — ")[0]
    dest_iata = dest_label.split(" — ")[0]

    if origin_iata == dest_iata:
        st.warning("Origin and destination must be different airports.")
        st.stop()

    airline_code = opts["airlines"][airline_idx][0]
    airline_display = opts["airlines"][airline_idx][1]

    with st.spinner("Running prediction…"):
        model, threshold = inference.load_model()
        lookups = inference.load_lookups()
        row = inference.build_feature_row(
            origin=origin_iata,
            dest=dest_iata,
            airline=airline_code,
            month=month,
            day_of_week=day_of_week,
            hour=hour,
            weather_label=weather_label,
            lookups=lookups,
            airports=airports_df,
        )
        prob, is_delayed = inference.predict_delay(row, model, threshold)

    pct = prob * 100
    thresh_pct = threshold * 100

    if is_delayed:
        verdict_emoji = "🔴"
        verdict_text = "Likely delayed (≥15 min)"
        verdict_color = "#dc2626"
    else:
        verdict_emoji = "🟢"
        verdict_text = "Likely on-time"
        verdict_color = "#16a34a"

    st.markdown(
        f"""
        <div class='result-card'>
          <div class='result-verdict' style='color:{verdict_color};'>
            {verdict_emoji} {verdict_text}
          </div>
          <div class='result-prob'>
            Probability of delay: <strong>{pct:.1f}%</strong>
            &nbsp;&nbsp;·&nbsp;&nbsp; threshold: {thresh_pct:.1f}%
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(min(prob, 1.0), text=f"Delay probability: {pct:.1f}%")

    st.markdown(
        f"""
        <div class='input-summary'>
          <strong>Inputs:</strong> {airline_display} · {origin_iata} → {dest_iata} ·
          {DOW_LABELS[day_of_week - 1]} in {calendar.month_name[month]} at {hour:02d}:00 ·
          weather: {weather_label}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Engineered features (rolling delays, route-level rates, congestion, previous-leg) "
    "are static historical averages, serving as a proof-of-concept stand-in for what a "
    "streaming feature store would compute in a live deployment."
)
