from __future__ import annotations

"""Overall-network callbacks + selection details."""

from dash import Dash, Input, Output, State, html

from .config import DEFAULT_SIM_THRESHOLD, EDGE_WIDTH_MAX, EDGE_WIDTH_MIN
from .data_store import corpus, kw_index
from .graph_store import get_elements
from .vis_elements import bipartite_column_positions


def register(app: Dash) -> None:
    @app.callback(
        Output("overall-cyto", "elements"),
        Output("overall-cyto", "layout"),
        Output("overall-bipartite-headers", "style"),
        Output("overall-meta", "children"),
        Input("overall-network", "value"),
        Input("overall-min-value", "value"),
        Input("overall-node-scaling", "value"),
        Input("overall-edge-scaling", "value"),
        Input("overall-community", "value"),
        Input("overall-sim-threshold", "value"),
        prevent_initial_call=False,
    )
    def update_overall(
        network_kind, min_value, node_scaling, edge_scaling, community, sim_threshold
    ):
        elements = get_elements(
            network_kind,
            None,
            min_value=int(min_value or 0),
            sim_threshold=float(sim_threshold or DEFAULT_SIM_THRESHOLD),
            node_scaling=str(node_scaling),
            edge_scaling=str(edge_scaling),
            community=str(community),
        )

        is_bip = str(network_kind).startswith("concept_method")

        if is_bip:
            # Split node elements by node_type and order by weighted degree (falls back safely)
            node_els = [e for e in elements if "source" not in e.get("data", {})]
            concepts, methods = [], []
            for e in node_els:
                nt = (e.get("data", {}) or {}).get("node_type", "")
                if nt == "concept":
                    concepts.append(e)
                elif nt == "method":
                    methods.append(e)

            def score(e):
                d = e.get("data", {}) or {}
                return float(d.get("raw_weighted_degree") or d.get("raw_degree") or 0.0)

            concepts.sort(key=score, reverse=True)
            methods.sort(key=score, reverse=True)

            left_ids = [e["data"]["id"] for e in concepts]
            right_ids = [e["data"]["id"] for e in methods]

            pos = bipartite_column_positions(
                left_ids, right_ids, left_x=0.0, right_x=650.0, y_gap=42.0
            )

            # Inject positions + lock nodes
            for e in node_els:
                nid = e["data"]["id"]
                if nid in pos:
                    e["position"] = pos[nid]
                    e["locked"] = True  # prevent dragging

            layout = {"name": "preset", "fit": True, "padding": 20}
            headers_style = {
                "display": "flex",
                "justifyContent": "space-between",
                "padding": "0 18px 6px 18px",
                "fontWeight": 600,
                "color": "#444",
            }
        else:
            layout = {"name": "cose", "animate": True, "animationDuration": 650}
            headers_style = {"display": "none"}

        meta = [
            html.Div("Overall network"),
            html.Div(
                f"Nodes: {sum(1 for e in elements if 'source' not in e.get('data', {}))}  |  "
                f"Edges: {sum(1 for e in elements if 'source' in e.get('data', {}))}"
            ),
            html.Div(
                f"Edge width range: {EDGE_WIDTH_MIN}-{EDGE_WIDTH_MAX} (dashboard-scaled)"
            ),
        ]

        return elements, layout, headers_style, meta

    @app.callback(
        Output("overall-details", "children"),
        Input("overall-cyto", "tapNodeData"),
        Input("overall-cyto", "tapEdgeData"),
        State("overall-network", "value"),
        prevent_initial_call=False,
    )
    def show_overall_selection(node_data, edge_data, network_kind):
        if node_data is None and edge_data is None:
            return html.Div("Click a node or edge to see details.")

        if node_data is not None:
            kw = node_data.get("id", "")
            pubs = [corpus[i] for i in kw_index.get(kw, [])][:12]
            return html.Div(
                [
                    html.Div([html.B("Node:"), f" {kw}"]),
                    html.Div(
                        [html.B("node_type:"), f" {node_data.get('node_type','')}"]
                    ),
                    html.Div(
                        [html.B("community:"), f" {node_data.get('community','')}"]
                    ),
                    html.Hr(),
                    html.B("Publications containing this keyword (max 12):"),
                    (
                        html.Ul(
                            [
                                html.Li(
                                    [
                                        html.B(p.title),
                                        html.Div(
                                            f"{p.publication_date.isoformat()}  |  cited_by={p.total_cited_by}  |  source={p.source}"
                                        ),
                                    ],
                                    style={"marginBottom": "8px"},
                                )
                                for p in pubs
                            ]
                        )
                        if pubs
                        else html.Div(
                            "No publications found for this node (keyword normalization mismatch)."
                        )
                    ),
                ]
            )

        src = edge_data.get("source")
        tgt = edge_data.get("target")
        raw_w = edge_data.get("raw_weight")
        return html.Div(
            [
                html.Div([html.B("Edge:"), f" {src} — {tgt}"]),
                html.Div([html.B("raw_weight:"), f" {raw_w}"]),
            ]
        )
