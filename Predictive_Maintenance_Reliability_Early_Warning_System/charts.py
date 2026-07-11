# charts.py
import plotly.graph_objects as go

def get_risk_gauge(risk_score: int) -> go.Figure:
    """Creates an industrial-grade risk gauge with dynamic color states."""
    # Determine dynamic needle/bar color based on severity thresholds
    if risk_score > 75:
        bar_color = "#DC3545"      # Alert Red
    elif risk_score > 35:
        bar_color = "#FFC107"      # Warning Amber
    else:
        bar_color = "#0B3C5D"      # Normal Safe Industrial Blue

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        gauge={
            'axis': {
                'range': [0, 100], 
                'tickwidth': 1, 
                'tickcolor':"#495057",
                'tickfont': {'size': 14}
            },
            'bar': {'color': bar_color}, # Uses our dynamic calculation
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 1,
            'bordercolor': "#DEE2E6",
            'steps': [
                {'range': [0, 35], 'color': "rgba(40, 167, 69, 0.1)"},  # Green zone hue
                {'range': [35, 75], 'color': "rgba(255, 193, 7, 0.1)"}, # Orange zone hue
                {'range': [75, 100], 'color': "rgba(220, 53, 69, 0.1)"} # Red zone hue
            ]
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=260,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig