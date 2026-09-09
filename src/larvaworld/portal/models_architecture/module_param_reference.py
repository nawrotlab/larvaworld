"""Parameter reference table for Module Inspector - code to symbol mapping."""

from __future__ import annotations

__all__ = ["get_parameter_reference_html"]


def get_parameter_reference_html() -> str:
    """Generate HTML parameter reference table with symbols, units, and descriptions.

    Returns a searchable, sortable table showing:
    - Parameter code name (e.g., A_in)
    - Symbol (e.g., A<sub>in</sub>)
    - Data type
    - Typical range/default
    - Description
    """

    parameters = [
        {
            "code": "A_in",
            "symbol": "A<sub>in</sub>",
            "type": "float",
            "range": "[0, 1]",
            "description": "Input amplitude constant or stimulus scaling",
        },
        {
            "code": "freq",
            "symbol": "f",
            "type": "float",
            "range": "[0, 10]",
            "description": "Oscillation frequency (Hz)",
        },
        {
            "code": "A_coef",
            "symbol": "A<sub>coef</sub>",
            "type": "float",
            "range": "[0, 2]",
            "description": "Amplitude coefficient for modulation",
        },
        {
            "code": "d_mean",
            "symbol": "d̄",
            "type": "float",
            "range": "[0.01, 1]",
            "description": "Mean stride distance (body lengths)",
        },
        {
            "code": "d_std",
            "symbol": "σ<sub>d</sub>",
            "type": "float",
            "range": "[0, 0.5]",
            "description": "Stride distance standard deviation",
        },
        {
            "code": "sigma_out",
            "symbol": "σ<sub>out</sub>",
            "type": "float",
            "range": "[0, 0.2]",
            "description": "Output jitter/noise standard deviation",
        },
        {
            "code": "phase_0",
            "symbol": "φ<sub>0</sub>",
            "type": "float",
            "range": "[0, 2π]",
            "description": "Initial phase (radians)",
        },
    ]

    rows = "\n".join(
        f"""
        <tr>
            <td style='padding:6px;border-bottom:1px solid #eee;'><code>{p["code"]}</code></td>
            <td style='padding:6px;border-bottom:1px solid #eee;font-family:monospace;'>{p["symbol"]}</td>
            <td style='padding:6px;border-bottom:1px solid #eee;font-size:11px;'>{p["type"]}</td>
            <td style='padding:6px;border-bottom:1px solid #eee;font-size:11px;color:#666;'>{p["range"]}</td>
            <td style='padding:6px;border-bottom:1px solid #eee;font-size:12px;'>{p["description"]}</td>
        </tr>
        """
        for p in parameters
    )

    return f"""
    <div style='margin-top:10px;'>
        <table style='width:100%;border-collapse:collapse;font-size:12px;'>
            <thead style='background:rgba(0,0,0,0.05);border-bottom:2px solid #ccc;'>
                <tr>
                    <th style='padding:8px;text-align:left;font-weight:600;'>Code Name</th>
                    <th style='padding:8px;text-align:left;font-weight:600;'>Symbol</th>
                    <th style='padding:8px;text-align:left;font-weight:600;'>Type</th>
                    <th style='padding:8px;text-align:left;font-weight:600;'>Typical Range</th>
                    <th style='padding:8px;text-align:left;font-weight:600;'>Description</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>

    <div style='margin-top:12px;font-size:11px;color:#666;border-top:1px solid #ddd;padding-top:8px;'>
        <p><strong>Legend:</strong></p>
        <ul style='margin:6px 0;padding-left:20px;'>
            <li>Hover over symbols to see their representations</li>
            <li>Ranges shown are typical for crawler/turner modules</li>
            <li>Sensor modules may have different ranges and additional parameters</li>
        </ul>
    </div>
    """
