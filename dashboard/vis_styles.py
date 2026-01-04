from __future__ import annotations

"""Dash Cytoscape stylesheet.

Keep all styling changes in this file to avoid accidental visual regressions elsewhere.
"""

CYTO_STYLESHEET = [
    # Base node style
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "width": "data(size)",
            "height": "data(size)",
            "background-color": "data(color)",
            "shape": "data(shape)",
            "font-size": 12,
            "text-wrap": "wrap",
            "text-max-width": 140,
            "border-width": 0,
            "transition-property": "opacity, border-width, border-color, width, height",
            "transition-duration": "520ms",
            "opacity": 1,
        },
    },
    {
        "selector": "node:selected",
        "style": {
            "border-width": 2,
            "border-color": "rgb(60,60,60)",
        },
    },
    {
        "selector": "edge",
        "style": {
            "width": "data(width)",
            "curve-style": "bezier",
            "opacity": 0.55,
            "transition-property": "opacity, width",
            "transition-duration": "520ms",
        },
    },
    {
        "selector": "edge:selected",
        "style": {
            "opacity": 0.9,
        },
    },
    {"selector": ".inactive", "style": {"opacity": 0.06, "label": "", "events": "no"}},
    {"selector": ".inactive-edge", "style": {"opacity": 0.05, "events": "no"}},
    {"selector": ".incoming", "style": {"opacity": 0.06}},
    {"selector": ".incoming-edge", "style": {"opacity": 0.05}},
    {"selector": ".birth", "style": {"border-width": 0}},
    {"selector": ".death", "style": {"border-width": 0}},
]
