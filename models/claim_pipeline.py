from PIL import Image

from models.vehicle_detector import analyze_damage, detect_fraud_indicators


SEVERITY_COST_RANGES = {
    "Minor": (2000, 5000),
    "Moderate": (5000, 20000),
    "Severe": (20000, 45000),
    "Total Loss": (60000, 150000),
}

AREA_MULTIPLIERS = {
    "front bumper": 1.15,
    "hood": 1.2,
    "door": 1.18,
    "headlight": 1.12,
    "side panel": 1.16,
}

BRAND_MULTIPLIERS = {
    "maruti": 1.0,
    "suzuki": 1.0,
    "hyundai": 1.08,
    "honda": 1.1,
    "toyota": 1.12,
    "tata": 1.06,
    "mahindra": 1.08,
    "kia": 1.1,
    "volkswagen": 1.2,
    "skoda": 1.22,
    "ford": 1.14,
    "nissan": 1.12,
    "renault": 1.1,
    "mg": 1.15,
    "jeep": 1.28,
    "audi": 1.55,
    "bmw": 1.65,
    "mercedes": 1.7,
    "benz": 1.7,
    "volvo": 1.58,
    "porsche": 1.9,
}

VEHICLE_MULTIPLIERS = {
    "bike": 0.75,
    "motorcycle": 0.75,
    "scooter": 0.75,
    "scooty": 0.75,
    "car": 1.0,
    "truck": 1.4,
    "bus": 1.5,
}


def infer_vehicle_type_from_text(description):
    """Infer the vehicle type from user-provided text when possible."""
    desc_lower = (description or "").lower()

    keyword_groups = [
        (("scooty", "scooter", "motor bike", "motorbike", "motor cycle", "motorcycle", "bike"), "bike"),
        (("truck", "lorry", "pickup"), "truck"),
        (("bus",), "bus"),
        (("car", "sedan", "hatchback", "suv", "jeep"), "car"),
    ]

    for keywords, vehicle_type in keyword_groups:
        if any(keyword in desc_lower for keyword in keywords):
            return vehicle_type

    return None


def analyze_claim(image_path, description, inferred_vehicle_type=None):
    """
    Process a claim according to the requested workflow.

    Returns:
        dict: structured claim analysis for persistence and UI display.
    """
    try:
        damage_analysis = analyze_damage(image_path)
        if "error" in damage_analysis:
            return _fallback_analysis(image_path, description, inferred_vehicle_type=inferred_vehicle_type)

        vehicle_type = _resolve_vehicle_type(inferred_vehicle_type or damage_analysis.get("vehicle_type"), description)
        damage_detected = damage_analysis.get("damage_detected", False)
        damage_areas = damage_analysis.get("damage_areas", [])
        severity = _resolve_severity(damage_analysis)

        if not damage_detected and _has_visible_damage_evidence(damage_analysis):
            damage_detected = True

        if not damage_detected:
            fraud_analysis = detect_fraud_indicators(
                image_path,
                description,
                damage_analysis=damage_analysis,
                estimated_amount=0,
            )
            fraud_score = fraud_analysis.get("fraud_score", 0.0)
            mismatch = _check_mismatch(fraud_analysis)
            return {
                "vehicle_valid": True,
                "vehicle_type": vehicle_type,
                "status": "needs_review" if mismatch else "no_damage",
                "severity": "None",
                "damage_severity": "None",
                "damage_description": "Damage: None",
                "damage_summary": "Damage: None",
                "damage_areas": [],
                "damage_location": "None",
                "fraud_score": fraud_score,
                "fraud_analysis": fraud_analysis,
                "mismatch": mismatch,
                "estimated_amount": 0.0,
                "repair_recommendation": "No repair required. No visible damage detected.",
                "parts_to_replace": "None",
            }

        estimated_amount = _calculate_cost_estimate(
            severity,
            damage_areas,
            description,
            vehicle_type,
        )

        fraud_analysis = detect_fraud_indicators(
            image_path,
            description,
            damage_analysis=damage_analysis,
            estimated_amount=estimated_amount,
        )
        fraud_score = fraud_analysis.get("fraud_score", 0.0)
        mismatch = _check_mismatch(fraud_analysis)

        damage_description = _compose_damage_description(damage_analysis)
        repair_recommendation, parts_to_replace = _build_repair_recommendation(
            damage_areas,
            severity,
        )

        return {
            "vehicle_valid": True,
            "vehicle_type": vehicle_type,
            "status": _resolve_claim_status(severity, fraud_score, mismatch),
            "severity": severity,
            "damage_severity": severity,
            "damage_description": damage_description,
            "damage_summary": damage_description,
            "damage_areas": damage_areas,
            "damage_location": ", ".join(damage_areas) if damage_areas else "None",
            "fraud_score": fraud_score,
            "fraud_analysis": fraud_analysis,
            "mismatch": mismatch,
            "estimated_amount": estimated_amount,
            "repair_recommendation": repair_recommendation,
            "parts_to_replace": parts_to_replace,
        }

    except Exception as exc:
        print(f"Error in claim pipeline: {exc}")
        return _fallback_analysis(image_path, description, inferred_vehicle_type=inferred_vehicle_type)


def _normalize_vehicle_type(vehicle_type):
    """Convert detector labels into user-facing values."""
    if not vehicle_type:
        return None

    normalized = str(vehicle_type).lower().strip().replace("_", " ").replace("-", " ")

    if normalized in {"motorcycle", "motor bike", "motorbike", "motor cycle", "bike", "scooter", "scooty"}:
        return "bike"
    if normalized in {"car", "sedan", "hatchback", "suv", "jeep"}:
        return "car"
    if normalized in {"truck", "lorry", "pickup"}:
        return "truck"
    if normalized == "bus":
        return "bus"
    if normalized == "unknown":
        return "unknown"
    return normalized


def _resolve_vehicle_type(vehicle_type, description):
    """Prefer explicit user text over weak image-only vehicle guesses."""
    described_vehicle_type = infer_vehicle_type_from_text(description)
    normalized_vehicle_type = _normalize_vehicle_type(vehicle_type)

    if described_vehicle_type:
        return described_vehicle_type

    return normalized_vehicle_type or "unknown"


def _resolve_severity(damage_analysis):
    """Promote borderline evidence into a reviewable severity instead of 'None'."""
    severity = damage_analysis.get("severity_level", "None")
    damage_areas = damage_analysis.get("damage_areas", [])
    confidence = float(damage_analysis.get("confidence", 0.0) or 0.0)
    damage_extent = float(damage_analysis.get("damage_extent", 0.0) or 0.0)

    if severity == "None" and confidence >= 0.45 and damage_areas:
        return "Minor"

    if severity == "Minor" and len(damage_areas) >= 2 and confidence >= 0.45 and damage_extent >= 0.05:
        return "Moderate"

    return severity


def _has_visible_damage_evidence(damage_analysis):
    """Catch obvious-but-borderline damage evidence that the CV gate marked too conservatively."""
    confidence = float(damage_analysis.get("confidence", 0.0) or 0.0)
    severity_score = float(damage_analysis.get("severity_score", 0.0) or 0.0)
    damage_extent = float(damage_analysis.get("damage_extent", 0.0) or 0.0)
    damage_areas = damage_analysis.get("damage_areas", [])
    indicators_found = int(damage_analysis.get("indicators_found", 0) or 0)

    return (
        len(damage_areas) >= 2
        and indicators_found >= 2
        and confidence >= 0.45
        and (severity_score >= 0.25 or damage_extent >= 0.05)
    )


def _calculate_cost_estimate(severity, damage_areas, description, vehicle_type):
    """Estimate cost using severity range, area multipliers, and brand multiplier."""
    if severity == "None":
        return 0.0

    min_cost, max_cost = SEVERITY_COST_RANGES.get(severity, (5000, 20000))
    base_cost = (min_cost + max_cost) / 2
    if severity == "Severe":
        base_cost = max(base_cost, 24000)
    if severity == "Total Loss":
        base_cost = max(base_cost, 80000)

    area_multiplier = 1.0
    for area_name in damage_areas:
        area_multiplier *= AREA_MULTIPLIERS.get(area_name, 1.05)

    brand_multiplier = _extract_brand_multiplier(description)
    vehicle_multiplier = VEHICLE_MULTIPLIERS.get(vehicle_type, 1.0)

    final_cost = base_cost * area_multiplier * brand_multiplier * vehicle_multiplier
    return round(final_cost, 2)


def _extract_brand_multiplier(description):
    """Infer a brand multiplier from the text description."""
    desc_lower = description.lower()
    for brand_name, multiplier in BRAND_MULTIPLIERS.items():
        if brand_name in desc_lower:
            return multiplier
    return 1.0


def _compose_damage_description(damage_analysis):
    """Build the final damage description string."""
    damage_description = damage_analysis.get("damage_description")
    if damage_description and damage_description != "Damage: None":
        return damage_description

    areas = damage_analysis.get("damage_areas", [])
    if not areas:
        return "Damage: None"

    damage_character = damage_analysis.get("damage_character", "damaged")
    area_descriptions = [f"{area} {damage_character}" for area in areas[:3]]
    return ", ".join(area_descriptions).capitalize()


def _build_repair_recommendation(damage_areas, severity):
    """Generate repair advice from the detected areas and severity."""
    if severity == "None" or not damage_areas:
        return "No repair required. No visible damage detected.", "None"

    repair_actions = []
    replace_parts = []

    for area_name in damage_areas:
        if area_name == "front bumper":
            repair_actions.append("replace or realign the front bumper")
            replace_parts.append("front bumper")
        elif area_name == "hood":
            repair_actions.append("repair the hood and refinish the panel")
            replace_parts.append("hood")
        elif area_name == "door":
            repair_actions.append("repair the door skin and repaint the section")
            replace_parts.append("door")
        elif area_name == "headlight":
            repair_actions.append("replace the headlight assembly")
            replace_parts.append("headlight")
        elif area_name == "side panel":
            repair_actions.append("reshape the side panel and repaint it")
            replace_parts.append("side panel")

    if severity == "Minor":
        closing_note = "Minor repainting and polishing required."
    elif severity == "Moderate":
        closing_note = "Panel alignment and repainting required."
    elif severity == "Severe":
        closing_note = "Replacement of damaged parts and structural inspection required."
    else:
        closing_note = "Complete workshop inspection recommended before approving repairs."

    recommendation = f"{_join_phrases(repair_actions).capitalize()}. {closing_note}"
    return recommendation, ", ".join(dict.fromkeys(replace_parts))


def _join_phrases(items):
    """Join short phrases into natural language."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _check_mismatch(fraud_analysis):
    """Determine whether the description and image need manual review."""
    return (
        not fraud_analysis.get("location_consistent", True)
        or not fraud_analysis.get("severity_consistent", True)
    )


def _resolve_claim_status(severity, fraud_score, mismatch):
    """Map claim results to app statuses."""
    if fraud_score >= 0.75:
        return "fraud"
    if mismatch or severity == "Total Loss":
        return "needs_review"
    return "approved"


def _fallback_analysis(image_path, description, inferred_vehicle_type=None):
    """Safe fallback when the image pipeline cannot produce a detailed result."""
    vehicle_type = _resolve_vehicle_type(inferred_vehicle_type, description)

    try:
        Image.open(image_path)
        return {
            "vehicle_valid": True,
            "vehicle_type": vehicle_type,
            "status": "needs_review",
            "severity": "Unknown",
            "damage_severity": "Unknown",
            "damage_description": "Image uploaded, but automated damage analysis could not be completed.",
            "damage_summary": "Image uploaded, but automated damage analysis could not be completed.",
            "damage_areas": [],
            "damage_location": "Unknown",
            "fraud_score": 0.35 if len(description.strip()) < 20 else 0.15,
            "fraud_analysis": {"indicators": ["Automated analysis fallback was used."]},
            "mismatch": False,
            "estimated_amount": 0.0,
            "repair_recommendation": "Manual inspection required.",
            "parts_to_replace": "Unknown",
        }
    except Exception as exc:
        print(f"Fallback analysis error: {exc}")
        return {
            "vehicle_valid": True,
            "vehicle_type": vehicle_type,
            "status": "needs_review",
            "severity": "Unknown",
            "damage_severity": "Unknown",
            "damage_description": "Analysis failed.",
            "damage_summary": "Analysis failed.",
            "damage_areas": [],
            "damage_location": "Unknown",
            "fraud_score": 0.5,
            "fraud_analysis": {"indicators": ["Analysis failed."]},
            "mismatch": True,
            "estimated_amount": 0.0,
            "repair_recommendation": "Please try again with a clearer image.",
            "parts_to_replace": "None",
        }
