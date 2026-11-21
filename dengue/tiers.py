DEFAULT_THRESHOLDS = {
    "medium": 2.0,    # 2 to < 5 weekly cases
    "high": 5.0,      # 5 to < 10 weekly cases
    "critical": 10.0, # >= 10 weekly cases (top ~10% surge)
}

TIER_META = {
    "Low": {
        "color": "#2E7D32",
        "action": "Routine larval surveillance and standard municipal waste collection.",
    },
    "Medium": {
        "color": "#F9A825",
        "action": "Inspect water-logged hotspots and issue community source-reduction advisories.",
    },
    "High": {
        "color": "#EF6C00",
        "action": "Deploy targeted ward fogging, larvicide teams, and alert local clinics.",
    },
    "Critical": {
        "color": "#C62828",
        "action": "Activate emergency outbreak response, intensive vector control, and hospital surge beds.",
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