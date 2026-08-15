"""Mathematical equations display for Module Inspector."""

from __future__ import annotations

__all__ = ["get_module_equations_html"]


def _format_equation(label: str, equation: str, description: str = "") -> str:
    """Format a single equation with label, formula, and description."""
    desc_html = (
        f"<div style='font-size:11px;opacity:0.8;margin-top:2px;'>{description}</div>"
        if description
        else ""
    )
    return f"""
    <div style='margin-bottom:12px;'>
        <div style='font-weight:600;font-size:13px;'>{label}</div>
        <div style='font-family:monospace;background:rgba(0,0,0,0.05);padding:6px;border-radius:4px;margin:4px 0;'>{equation}</div>
        {desc_html}
    </div>
    """


def _common_equations_html() -> str:
    """Equations applicable to all crawler/turner modules."""
    return f"""
    <div style='border-left:3px solid #4a7c9e;padding-left:10px;'>
        <h4 style='margin:8px 0;color:#333;'>Common Foundation (All Effectors)</h4>

        {_format_equation(
            "Phase Evolution",
            f"φ(t) = 2πft mod 2π",
            "Phase advances with frequency f, cycling every 1/f seconds"
        )}

        {_format_equation(
            "Stride Distance (Stochastic)",
            f"d_n(t) ~ N(d_mean, d_std²)",
            "Stride distance resampled per cycle with mean and std from parameters"
        )}

        {_format_equation(
            "Motor Output with Noise",
            f"Act_out(t) = Act(t) · (1 + η_out)",
            "where η_out ~ N(0, σ_out²) is output jitter"
        )}

        {_format_equation(
            "General Form",
            f"Act(t) = f · d_n(t) · [1 + A_coef · M(φ(t))]",
            "Frequency × stride × modulation: f=freq, d_n=stride, A_coef=amplitude, M(φ)=mode-specific"
        )}
    </div>
    """


def _crawler_modes_html() -> str:
    """Mode-specific equations for crawler effector."""
    return f"""
    <div style='border-left:3px solid #6b9e7c;padding-left:10px;'>
        <h4 style='margin:8px 0;color:#333;'>Crawler Modulation Functions M(φ)</h4>

        {_format_equation(
            "Constant Mode",
            f"M(φ) = 1.0",
            "No modulation; constant activation"
        )}

        {_format_equation(
            "Gaussian Mode",
            f"M(φ) = exp(−((φ·180/π − 180)² / (2σ_cycle²)))",
            "Gaussian envelope centered at phase 180°; σ_cycle controls width"
        )}

        {_format_equation(
            "Square Mode",
            f"M(φ) = {'{+1 if φ < 2πd; −1 otherwise}'}",
            "Duty-cycle modulation; d parameter controls on/off duration"
        )}

        {_format_equation(
            "Phase Oscillator Mode",
            f"M(φ) = cos(φ − φ_max)",
            "Cosine modulation with peak at φ_max"
        )}
    </div>
    """


def _sensor_equations_html() -> str:
    """Equations for sensor modules."""
    return f"""
    <div style='border-left:3px solid #9e7c6b;padding-left:10px;'>
        <h4 style='margin:8px 0;color:#333;'>Sensor Input-Output</h4>

        {_format_equation(
            "Derivative-Based Response",
            f"output = gain · d(stimulus)/dt",
            "Sensors respond to stimulus change, not absolute value"
        )}

        {_format_equation(
            "Frequency Filtering",
            f"output ∝ F(f) · stimulus(f)",
            "Frequency response F(f) shapes sensitivity across frequencies"
        )}

        {_format_equation(
            "Attenuation (Some Sensors)",
            f"A(t) = A_0 · (1 − e^(−t/τ))",
            "Temporal adaptation with time constant τ"
        )}
    </div>
    """


def get_module_equations_html(module_id: str, mode: str = "") -> str:
    """Get equations HTML for a module and optional mode."""
    html = """
    <div style='font-family:system-ui,sans-serif;font-size:12px;line-height:1.5;'>
        <style>
            .equations-section { margin-bottom: 16px; }
            .equations-section h4 { margin: 0; padding-bottom: 4px; border-bottom: 1px solid #ddd; }
        </style>
    """

    # Common equations (always show for effectors)
    if module_id in ("crawler", "turner"):
        html += '<div class="equations-section">' + _common_equations_html() + "</div>"
        html += '<div class="equations-section">' + _crawler_modes_html() + "</div>"
    elif module_id == "feeder":
        html += """
        <div style='border-left:3px solid #9e9e4b;padding-left:10px;'>
            <h4 style='margin:8px 0;color:#333;'>Feeder (Self-Oscillator)</h4>
        """
        html += _format_equation(
            "Self-Sustained Oscillation",
            "φ(t) = 2πft (constant frequency, no external input)",
            "Feeder runs autonomously; A_in is ignored (mode: started)",
        )
        html += "</div>"
    elif module_id in ("olfactor", "toucher", "windsensor", "thermosensor"):
        html += '<div class="equations-section">' + _sensor_equations_html() + "</div>"

    html += "</div>"
    return html
