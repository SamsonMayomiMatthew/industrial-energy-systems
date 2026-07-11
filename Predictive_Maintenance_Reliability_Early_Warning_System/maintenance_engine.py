"""Maintenance Logic Engine for calculating risk scores and action triggers."""

from typing import Dict, Any
import pandas as pd
import numpy as np

def generate_maintenance_plan(failure_prob: float, confidence: float) -> Dict[str, Any]:
    """Translates failure probability into concrete maintenance workflows."""
    
    if failure_prob > 0.8:
        priority = "P1"
        action = "IMMEDIATE SHUTDOWN & INSPECTION"
        downtime = "4-8 Hours"
        team = "Emergency Maintenance"
    elif failure_prob > 0.4:
        priority = "P2"
        action = "SCHEDULED PREVENTATIVE MAINTENANCE"
        downtime = "1-2 Hours"
        team = "Day Shift Mechanical"
    else:
        priority = "P3"
        action = "MONITOR & RECORD DATA"
        downtime = "None"
        team = "Routine Monitoring"
        
    risk_score = int(failure_prob * 100)
    
    return {
        "Priority": priority,
        "Risk_Score": risk_score,
        "Recommended_Action": action,
        "Est_Downtime": downtime,
        "Team": team,
        "Confidence": f"{confidence * 100:.1f}%"
    }