import os
import random
from PIL import Image
from models.vehicle_detector import analyze_damage, detect_fraud_indicators

def analyze_claim(image_path, description, inferred_vehicle_type=None):
    """
    Advanced AI analysis using Vision-Language Model simulation.
    Now includes comprehensive vehicle detection, damage analysis, and fraud detection.

    Args:
        image_path (str): Path to the uploaded image
        description (str): User's description of the incident
        inferred_vehicle_type (str, optional): Vehicle type detected before analysis.

    Returns:
        tuple: (severity, mismatch, fraud_score, estimated_amount, damage_summary,
                repair_recommendation, parts_to_replace, vehicle_type, damage_location,
                damage_severity, fraud_analysis)
    """
    try:
        # Step 1: Advanced damage analysis
        damage_analysis = analyze_damage(image_path)

        if "error" in damage_analysis:
            return ('unknown', True, 0.3, 0, 'Image analysis failed. Please try uploading the image again or ensure it\'s a valid image format (JPG, PNG).',
                   'Unable to analyze image - please provide a clear photo of the vehicle.', 'None', 'unknown', 'unknown', 'unknown', {})

        # Step 2: Fraud detection
        fraud_analysis = detect_fraud_indicators(image_path, description)

        if "error" in fraud_analysis:
            fraud_score = 0.5  # Default moderate fraud score
        else:
            fraud_score = fraud_analysis.get("fraud_score", 0.5)

        # Step 3: Determine severity based on multiple factors
        severity = _determine_severity(description, damage_analysis, fraud_analysis)

        # Step 4: Check for mismatch between description and image analysis
        mismatch = _check_description_image_mismatch(description, damage_analysis, fraud_analysis)

        # Step 5: Enhanced cost estimation
        estimated_amount = _calculate_detailed_cost_estimate(
            severity, description, damage_analysis, fraud_score
        )

        # Step 6: Generate comprehensive damage summary
        damage_summary = _generate_damage_summary(
            damage_analysis, description, severity
        )

        # Step 7: Generate repair recommendations
        repair_recommendation, parts_to_replace = _generate_repair_recommendations(
            damage_analysis, severity, estimated_amount
        )

        # Step 8: Extract additional analysis results
        vehicle_type = inferred_vehicle_type or damage_analysis.get("vehicle_type") or "unknown"
        damage_location = damage_analysis.get("damage_location", "unknown")
        damage_severity = damage_analysis.get("severity_level", severity)

        return (severity, mismatch, fraud_score, estimated_amount, damage_summary,
                repair_recommendation, parts_to_replace, vehicle_type, damage_location,
                damage_severity, fraud_analysis)

    except Exception as e:
        print(f"Error in advanced claim analysis: {e}")
        # Fallback to basic analysis
        return _basic_fallback_analysis(image_path, description)

def _determine_severity(description, damage_analysis, fraud_analysis):
    """Determine claim severity based on multiple analysis sources."""
    desc_lower = description.lower()

    # Severity from description keywords
    desc_severity = 'medium'
    if any(word in desc_lower for word in ['minor', 'small', 'scratch', 'dent']):
        desc_severity = 'low'
    elif any(word in desc_lower for word in ['major', 'severe', 'extensive', 'totaled']):
        desc_severity = 'high'

    # Severity from image analysis
    image_severity = damage_analysis.get("severity_level", "medium")

    # Map string severities to numeric for averaging
    severity_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
    desc_score = severity_map.get(desc_severity, 2)
    image_score = severity_map.get(image_severity, 2)

    # Weighted average (description 40%, image 60%)
    combined_score = (desc_score * 0.4 + image_score * 0.6)

    # Convert back to string
    if combined_score < 1.5:
        return 'low'
    elif combined_score < 2.5:
        return 'medium'
    else:
        return 'high'

def _check_description_image_mismatch(description, damage_analysis, fraud_analysis):
    """Check for mismatches between description and image analysis."""
    desc_lower = description.lower()

    # Check if description mentions damage
    damage_mentioned = any(word in desc_lower for word in
                          ['damage', 'accident', 'collision', 'crash', 'hit', 'dent', 'scratch'])

    # Check if image shows damage
    damage_detected = damage_analysis.get("damage_detected", False)

    # Mismatch if description claims damage but image shows none (or vice versa)
    if damage_mentioned and not damage_detected:
        return True
    if not damage_mentioned and damage_detected:
        return True

    # Check location consistency
    location_consistent = fraud_analysis.get("location_consistent", True)
    if not location_consistent:
        return True

    return False

def _calculate_detailed_cost_estimate(severity, description, damage_analysis, fraud_score):
    """Calculate detailed cost estimate based on multiple factors."""
    desc_lower = description.lower()

    # Base amount by severity
    base_amounts = {'low': 1200, 'medium': 2400, 'high': 4800}
    base_amount = base_amounts.get(severity, 2400)

    # Damage type multipliers
    damage_type = damage_analysis.get("damage_type", "unknown")
    type_multipliers = {
        'paint_damage': 1.2,
        'structural_damage': 2.5,
        'dent_damage': 1.8,
        'surface_damage': 1.1,
        'unknown_damage': 1.5
    }
    base_amount *= type_multipliers.get(damage_type, 1.5)

    # Part-specific costs
    parts_cost = 0
    part_costs = {
        'bumper': 450, 'hood': 520, 'headlight': 320, 'door': 420,
        'fender': 380, 'trunk': 480, 'roof': 600, 'windshield': 280,
        'tire': 230, 'mirror': 180, 'grille': 350
    }

    for part, cost in part_costs.items():
        if part in desc_lower:
            parts_cost += cost

    # Location-based adjustments
    damage_location = damage_analysis.get("damage_location", "unknown")
    location_multipliers = {
        'front': 1.3,  # Front damage often more expensive
        'rear': 1.1,
        'left': 1.0, 'right': 1.0,
        'roof': 1.4,  # Roof damage complex
        'multiple_areas': 2.0
    }
    location_multiplier = location_multipliers.get(damage_location, 1.0)

    # Fraud score adjustment (higher fraud = lower estimate)
    fraud_multiplier = max(0.5, 1.0 - fraud_score * 0.3)

    # Calculate final amount
    estimated_amount = (base_amount + parts_cost) * location_multiplier * fraud_multiplier

    # Add some randomization for realism (±10%)
    variation = random.uniform(0.9, 1.1)
    estimated_amount *= variation

    return round(estimated_amount, 2)

def _generate_damage_summary(damage_analysis, description, severity):
    """Generate comprehensive damage summary."""
    damage_detected = damage_analysis.get("damage_detected", False)
    damage_location = damage_analysis.get("damage_location", "unknown")
    damage_type = damage_analysis.get("damage_type", "unknown")
    confidence = damage_analysis.get("confidence", 0.0)

    if not damage_detected:
        return f"AI analysis shows no visible damage in the image. Description: {description[:100]}..."

    summary = f"AI analysis detected {severity} severity damage"

    if damage_location != "unknown":
        summary += f" primarily located at the {damage_location.replace('_', ' ')}"

    if damage_type != "unknown":
        type_descriptions = {
            'paint_damage': 'paint and finish damage',
            'structural_damage': 'structural and body panel damage',
            'dent_damage': 'dent and impact damage',
            'surface_damage': 'surface and cosmetic damage'
        }
        summary += f" with {type_descriptions.get(damage_type, damage_type)}"

    summary += f". Analysis confidence: {confidence:.1%}"

    return summary

def _generate_repair_recommendations(damage_analysis, severity, estimated_amount):
    """Generate detailed repair recommendations."""
    damage_type = damage_analysis.get("damage_type", "unknown")
    damage_location = damage_analysis.get("damage_location", "unknown")

    # Base recommendations
    recommendations = []

    if severity == 'low':
        recommendations.append("Minor cosmetic repairs and paint touch-up")
    elif severity == 'medium':
        recommendations.append("Panel replacement and structural repairs")
    else:
        recommendations.append("Major bodywork and structural reconstruction")

    # Type-specific recommendations
    type_recommendations = {
        'paint_damage': "Paint matching and refinishing required",
        'structural_damage': "Frame inspection and alignment needed",
        'dent_damage': "Paintless dent repair or panel replacement",
        'surface_damage': "Sanding, priming, and repainting"
    }

    if damage_type in type_recommendations:
        recommendations.append(type_recommendations[damage_type])

    # Location-specific advice
    location_advice = {
        'front': "Check radiator, cooling system, and airbags",
        'rear': "Inspect fuel system and rear suspension",
        'roof': "Verify roof integrity and rain sealing",
        'multiple_areas': "Complete vehicle inspection recommended"
    }

    if damage_location in location_advice:
        recommendations.append(location_advice[damage_location])

    repair_recommendation = ". ".join(recommendations)
    repair_recommendation += f". Estimated repair cost: ₹{estimated_amount:,.0f}"

    # Determine parts to replace
    parts_to_replace = _identify_parts_to_replace(damage_analysis, severity)

    return repair_recommendation, parts_to_replace

def _identify_parts_to_replace(damage_analysis, severity):
    """Identify which parts need replacement."""
    damage_type = damage_analysis.get("damage_type", "unknown")
    damage_location = damage_analysis.get("damage_location", "unknown")

    parts = []

    # Severity-based parts
    if severity == 'high':
        parts.extend(['body panels', 'structural components'])
    elif severity == 'medium':
        parts.extend(['damaged panels'])

    # Type-based parts
    type_parts = {
        'paint_damage': ['paint', 'clear coat'],
        'structural_damage': ['frame rails', 'body panels'],
        'dent_damage': ['body panels', 'reinforcements'],
        'surface_damage': ['surface materials']
    }

    if damage_type in type_parts:
        parts.extend(type_parts[damage_type])

    # Location-based parts
    location_parts = {
        'front': ['bumper', 'hood', 'headlights'],
        'rear': ['trunk', 'taillights', 'bumper'],
        'left': ['driver door', 'fender'],
        'right': ['passenger door', 'fender'],
        'roof': ['roof panel']
    }

    if damage_location in location_parts:
        parts.extend(location_parts[damage_location])

    # Remove duplicates and join
    unique_parts = list(set(parts))
    return ', '.join(unique_parts) if unique_parts else 'None required'

def _basic_fallback_analysis(image_path, description):
    """Fallback analysis in case of errors."""
    try:
        # Basic image check
        img = Image.open(image_path)
        width, height = img.size

        # Simple severity detection
        desc_lower = description.lower()
        if 'minor' in desc_lower or 'small' in desc_lower:
            severity = 'low'
        elif 'major' in desc_lower or 'severe' in desc_lower:
            severity = 'high'
        else:
            severity = 'medium'

        mismatch = 'damage' not in desc_lower and 'accident' not in desc_lower

        # Basic fraud score
        fraud_score = 0.0
        if len(description) < 15:
            fraud_score += 0.3
        fraud_score += random.uniform(0, 0.2)
        fraud_score = min(fraud_score, 1.0)

        # Basic cost estimate
        base_amount = 1200
        if severity == 'low':
            estimated_amount = base_amount * 0.8
        elif severity == 'medium':
            estimated_amount = base_amount * 1.5
        else:
            estimated_amount = base_amount * 2.8

        damage_summary = f"Basic analysis shows {severity} damage. Limited AI processing available."
        repair_recommendation = f"Recommended repairs for {severity} damage. Estimated cost: ₹{estimated_amount:,.0f}"
        parts_to_replace = "Various parts"

        return (severity, mismatch, fraud_score, estimated_amount, damage_summary,
                repair_recommendation, parts_to_replace, 'unknown', 'unknown', 'unknown', {})

    except Exception as e:
        print(f"Fallback analysis error: {e}")
        return ('unknown', True, 0.5, 0, 'Analysis failed.', 'Please try again.', 'None',
                'unknown', 'unknown', 'unknown', {})