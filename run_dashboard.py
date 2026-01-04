"""Run the Dash thesis dashboard.

This is the ONLY 'connector' script:
- creates the Dash app
- sets layout
- registers callbacks
- runs the server

Run from the repo root (where `analysis/` and `data/` packages live):
    python3 run_dashboard.py
"""

import dash_cytoscape as cyto
from dash import Dash

cyto.load_extra_layouts()

from dashboard.callbacks_overall import register as register_overall
from dashboard.callbacks_stats import register as register_stats
from dashboard.callbacks_temporal import register as register_temporal
from dashboard.ui_layout import make_app_layout, register_tab_router


def create_app() -> Dash:
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.layout = make_app_layout()

    register_tab_router(app)
    register_temporal(app)
    register_overall(app)
    register_stats(app)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
