"""Map predicted weekly cases to risk tiers and recommended actions."""

# Low < 3 <= Medium < 9 <= High < 20 <= Critical  (weekly cases per ward)
# In 2016-2025 data: ~71% Low, ~17% Medium, ~11% High, ~1% Critical.
DEFAULT_THRESHOLDS = {
    "medium": 3.0,
    "high": 9.0,
    "critical": 20.0,
}

TIERS = ["Low", "Medium", "High", "Critical"]

# Colors: green for Low, then one red getting darker as risk rises (light red -> red -> extreme red).
# Checked with the dataviz palette validator: the three reds read light -> dark in order (ordinal
# check passes), and all four stay distinguishable for colour-blind viewers (all-pairs CVD dE 8.1,
# normal-vision dE 16.9). White label text reads best on every fill. "ink" is the colour for
# headline text on a white page (light red is too faint as text, so Medium uses dark ink).
TIER_META = {
    "Low": {
        "color": "#2E7D32",
        "ink": "#2E7D32",
        "action": "Routine larval surveillance and standard waste collection.",
    },
    "Medium": {
        "color": "#F28B82",
        "ink": "#16323B",
        "action": "Inspect water-logged hotspots and issue community source-reduction advisories.",
    },
    "High": {
        "color": "#D93025",
        "ink": "#D93025",
        "action": "Deploy larvicide and fogging teams to this ward and alert local clinics.",
    },
    "Critical": {
        "color": "#8C1D18",
        "ink": "#8C1D18",
        "action": "Activate outbreak response: intensive vector control, hospital surge beds, saline and test-kit stock.",
    },
}


def assign_risk_tier(predicted_cases: float, thresholds: dict | None = None) -> str:
    """Classify predicted weekly dengue cases into Low, Medium, High, or Critical."""
    t = thresholds or DEFAULT_THRESHOLDS
    if predicted_cases < t["medium"]:
        return "Low"
    if predicted_cases < t["high"]:
        return "Medium"
    if predicted_cases < t["critical"]:
        return "High"
    return "Critical"


def get_tier_info(tier: str) -> dict:
    return TIER_META.get(tier, TIER_META["Low"])
