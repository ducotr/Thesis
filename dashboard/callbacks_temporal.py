from __future__ import annotations

"""Temporal callbacks (temporal_controller + temporal selection details).

This module should be the *only* place you touch when debugging temporal transitions.
"""

import random

import networkx as nx
from dash import Dash, Input, Output, State, callback_context, html, no_update

from .config import DEFAULT_SIM_THRESHOLD
from .data_store import corpus, kw_index
from .graph_store import get_graph
from .temporal_helpers import (
    BIRTH_STEPS,
    FADE_STEPS,
    MAX_MOVE_PER_TRANSITION,
    MOVE_STEPS,
    POS_SCALE,
    SPRING_ITER,
    SPRING_SEED,
    _edge_key,
    _neighbors_centroid,
    _pos_get,
    _pos_set,
    _smoothstep,
    align_target_to_start,
    spring_layout_cyto,
)
from .vis_elements import (
    _community_color_map,
    bipartite_column_positions,
    edge_width_map,
)


def register(app: Dash) -> None:
    @app.callback(
        Output("temp-cyto", "elements"),
        Output("temp-meta", "children"),
        Output("temp-interval", "disabled"),
        Output("temp-current-year", "data"),
        Output("temp-pos-store", "data"),
        Output("temp-transition-store", "data"),
        Output("temp-cyto", "zoom"),
        Output("temp-cyto", "pan"),
        Input("temp-interval", "n_intervals"),
        Input("temp-network", "value"),
        Input("temp-year", "value"),
        Input("temp-min-value", "value"),
        Input("temp-node-scaling", "value"),
        Input("temp-edge-scaling", "value"),
        Input("temp-community", "value"),
        Input("temp-sim-threshold", "value"),
        State("temp-current-year", "data"),
        State("temp-pos-store", "data"),
        State("temp-transition-store", "data"),
        State("temp-cyto", "zoom"),
        State("temp-cyto", "pan"),
        prevent_initial_call=False,
    )
    def temporal_controller(
        _tick,
        network_kind,
        year,
        min_value,
        node_scaling,
        edge_scaling,
        community,
        sim_threshold,
        current_year,
        pos_store,
        transition,
        cur_zoom,
        cur_pan,
    ):
        """
        Staged temporal transition:
          1) Persisting nodes stay put while deaths fade out (no movement)
          2) Births appear & fade in near neighbors (still no movement)
          3) Then ALL remaining nodes gently settle into a cleaner layout
        """
        pos_store = pos_store or {}
        year = int(year)
        min_value = int(min_value or 0)
        sim_threshold = float(sim_threshold or DEFAULT_SIM_THRESHOLD)
        node_scaling = str(node_scaling)
        edge_scaling = str(edge_scaling)
        community = str(community)

        trig = callback_context.triggered_id

        # -----------------------------
        # Smooth camera follow (fit bbox + lerp zoom/pan)
        # -----------------------------
        VIEW_W = 1050  # approx px of cyto area; tweak if needed
        VIEW_H = 650  # should match cyto height in ui_layout
        PAD = 50
        MIN_ZOOM = 0.20
        MAX_ZOOM = 3.00
        SMOOTH_ALPHA = 0.25  # 0.10 slower/smoother, 0.25 snappier

        def _bbox(nodes: set[str]):
            xs, ys = [], []
            for n in nodes:
                p = _pos_get(pos_store, n)
                if p is None:
                    continue
                xs.append(p[0])
                ys.append(p[1])
            if not xs:
                return None
            return min(xs), min(ys), max(xs), max(ys)

        def _fit_zoom_pan(nodes: set[str]):
            bb = _bbox(nodes)
            if bb is None:
                return None
            xmin, ymin, xmax, ymax = bb
            w = max(1e-6, xmax - xmin)
            h = max(1e-6, ymax - ymin)
            zoom = min((VIEW_W - 2 * PAD) / w, (VIEW_H - 2 * PAD) / h)
            zoom = max(MIN_ZOOM, min(MAX_ZOOM, zoom))
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            pan = {
                "x": (VIEW_W / 2.0) - zoom * cx,
                "y": (VIEW_H / 2.0) - zoom * cy,
            }
            return zoom, pan

        def _camera(nodes: set[str]):
            # If user changed controls (not interval), fit directly.
            force_direct = trig != "temp-interval"
            target = _fit_zoom_pan(nodes)
            if target is None:
                return no_update, no_update
            tz, tp = target

            if force_direct or cur_zoom is None or cur_pan is None:
                return tz, tp

            try:
                cz = float(cur_zoom)
                cpx = float((cur_pan or {}).get("x", 0.0))
                cpy = float((cur_pan or {}).get("y", 0.0))
            except Exception:
                return tz, tp

            nz = cz + SMOOTH_ALPHA * (tz - cz)
            np = {
                "x": cpx + SMOOTH_ALPHA * (tp["x"] - cpx),
                "y": cpy + SMOOTH_ALPHA * (tp["y"] - cpy),
            }
            return nz, np

        def _ret(
            elements,
            meta,
            disabled,
            year_out,
            pos_store_out,
            transition_out,
            cam_nodes: set[str],
        ):
            if not cam_nodes:
                return (
                    elements,
                    meta,
                    disabled,
                    year_out,
                    pos_store_out,
                    transition_out,
                    no_update,
                    no_update,
                )
            z, p = _camera(cam_nodes)
            return (
                elements,
                meta,
                disabled,
                year_out,
                pos_store_out,
                transition_out,
                z,
                p,
            )

        def build_node_el(
            nid: str,
            data: dict,
            color_map: dict,
            cls: str,
            locked_node_colors: dict[str, str] | None = None,
        ) -> dict:
            size = float(data.get(node_scaling, 18.0))
            comm = data.get(community, 0)
            color = (
                locked_node_colors.get(nid)
                if locked_node_colors and nid in locked_node_colors
                else color_map.get(comm, "rgb(160,160,160)")
            )
            node_type = data.get("node_type")
            p = _pos_get(pos_store, nid) or (0.0, 0.0)
            return {
                "data": {
                    "id": nid,
                    "label": nid,
                    "size": size,
                    "community": comm,
                    "color": color,
                    "node_type": node_type or "",
                },
                "position": {"x": p[0], "y": p[1]},
                "classes": cls,
            }

        def render_graph(
            G: nx.Graph,
            active_nodes: set[str],
            node_classes: dict[str, str],
            edge_classes: dict[tuple[str, str], str] | None = None,
            locked_node_colors: dict[str, str] | None = None,
        ) -> list[dict]:
            edge_classes = edge_classes or {}
            elements: list[dict] = []
            color_map = {} if locked_node_colors else _community_color_map(G, community)
            ew = edge_width_map(G, edge_scaling)

            for node, data in G.nodes(data=True):
                nid = str(node)
                if nid not in active_nodes:
                    continue
                elements.append(
                    build_node_el(
                        nid,
                        data,
                        color_map,
                        node_classes.get(nid, ""),
                        locked_node_colors=locked_node_colors,
                    )
                )

            for u, v, data in G.edges(data=True):
                su, sv = str(u), str(v)
                if su not in active_nodes or sv not in active_nodes:
                    continue
                width = ew.get((u, v), ew.get((v, u), 1.0))
                cls = edge_classes.get(_edge_key(su, sv), "")
                elements.append(
                    {
                        "data": {
                            "id": f"{su}__{sv}",
                            "source": su,
                            "target": sv,
                            "width": float(width),
                        },
                        "classes": cls,
                    }
                )

            return elements

        # -------------------------------------------------
        # Bipartite concept-method: show instantly in 2 columns (no staged animation)
        # -------------------------------------------------
        if str(network_kind).startswith("concept_method"):
            G_target = get_graph(
                network_kind, year, min_value=min_value, sim_threshold=sim_threshold
            )
            target_nodes = {str(n) for n in G_target.nodes()}

            if not target_nodes:
                meta = [
                    html.Div(f"Year: {year} (bipartite; empty)"),
                    html.Div("Nodes: 0 | Edges: 0"),
                ]
                return _ret([], meta, True, year, {}, None, set())

            # build ordered left/right lists
            concepts, methods = [], []
            for n in target_nodes:
                nt = (G_target.nodes.get(n, {}) or {}).get("node_type", "")
                if nt == "concept":
                    concepts.append(n)
                elif nt == "method":
                    methods.append(n)

            def score(nid: str) -> float:
                d = G_target.nodes.get(nid, {}) or {}
                return float(d.get("raw_weighted_degree") or d.get("raw_degree") or 0.0)

            concepts.sort(key=score, reverse=True)
            methods.sort(key=score, reverse=True)

            # overwrite positions in pos_store for a clean 2-column layout
            pos_store = {}
            pos = bipartite_column_positions(
                concepts, methods, left_x=0.0, right_x=650.0, y_gap=42.0
            )
            for nid, p in pos.items():
                _pos_set(pos_store, nid, p["x"], p["y"])

            node_classes = {n: "" for n in target_nodes}
            elements = render_graph(G_target, target_nodes, node_classes)

            meta = [
                html.Div(f"Year: {year} (bipartite layout)"),
                html.Div(
                    f"Nodes: {len(target_nodes)}  |  Edges: {G_target.number_of_edges()}"
                ),
            ]
            return _ret(elements, meta, True, year, pos_store, None, target_nodes)

        # -----------------------------
        # (Re)start transition if controls changed (not an interval tick),
        # or if there's no transition yet, or target year changed.
        # -----------------------------
        if (
            trig != "temp-interval"
            or not transition
            or int(transition.get("target_year", year)) != year
        ):
            # If we have no previous state (first interaction OR we just came from an empty graph),
            # don't animate from some default year — just show the requested year instantly.
            if not pos_store or current_year is None:
                G_target = get_graph(
                    network_kind, year, min_value=min_value, sim_threshold=sim_threshold
                )
                target_nodes = {str(n) for n in G_target.nodes()}

                # If the requested year is empty, show empty instantly.
                if len(target_nodes) == 0:
                    meta = [
                        html.Div(f"Year: {year} (instant; empty)"),
                        html.Div("Nodes: 0  |  Edges: 0"),
                    ]
                    return _ret([], meta, True, year, {}, None, set())
                # Otherwise, compute a stable initial layout and show instantly.
                init = nx.spring_layout(
                    G_target, seed=SPRING_SEED, iterations=SPRING_ITER, scale=1.0
                )
                for n, (x, y) in init.items():
                    _pos_set(pos_store, str(n), x * POS_SCALE, y * POS_SCALE)

                active_nodes = {str(n) for n in G_target.nodes()}
                node_classes = {n: "" for n in active_nodes}
                elements = render_graph(G_target, active_nodes, node_classes)

                meta = [
                    html.Div(f"Year: {year} (instant; no previous state)"),
                    html.Div(
                        f"Nodes: {len(active_nodes)}  |  Edges: {G_target.number_of_edges()}"
                    ),
                ]
                return _ret(elements, meta, True, year, pos_store, None, active_nodes)

            start_year = int(current_year)
            G_start = get_graph(
                network_kind,
                start_year,
                min_value=min_value,
                sim_threshold=sim_threshold,
            )
            G_target = get_graph(
                network_kind, year, min_value=min_value, sim_threshold=sim_threshold
            )

            start_nodes = {str(n) for n in G_start.nodes()}
            target_nodes = {str(n) for n in G_target.nodes()}

            # If the start network is empty, show the target instantly (no transition needed).
            if len(start_nodes) == 0:
                if len(target_nodes) == 0:
                    meta = [
                        html.Div(f"Year: {year} (instant; empty)"),
                        html.Div("Nodes: 0  |  Edges: 0"),
                    ]
                    return _ret([], meta, True, year, {}, None, set())

                # Layout the target once and show
                init = nx.spring_layout(
                    G_target, seed=SPRING_SEED, iterations=SPRING_ITER, scale=1.0
                )
                for n, (x, y) in init.items():
                    _pos_set(pos_store, str(n), x * POS_SCALE, y * POS_SCALE)

                active_nodes = {str(n) for n in G_target.nodes()}
                node_classes = {n: "" for n in active_nodes}
                elements = render_graph(G_target, active_nodes, node_classes)

                meta = [
                    html.Div(f"Year: {year} (instant; start empty)"),
                    html.Div(
                        f"Nodes: {len(active_nodes)}  |  Edges: {G_target.number_of_edges()}"
                    ),
                ]
                return _ret(elements, meta, True, year, pos_store, None, active_nodes)

            persistent = start_nodes & target_nodes
            deaths = start_nodes - target_nodes
            births = target_nodes - start_nodes

            # If the target network is empty, show empty instantly (no transition needed).
            if len(target_nodes) == 0:
                meta = [
                    html.Div(f"Year: {year} (instant; target empty)"),
                    html.Div("Nodes: 0  |  Edges: 0"),
                ]
                return _ret([], meta, True, year, {}, None, set())
            # Initialise positions once
            if G_start.number_of_nodes() and not pos_store:
                init = nx.spring_layout(
                    G_start, seed=SPRING_SEED, iterations=SPRING_ITER, scale=1.0
                )
                for n, (x, y) in init.items():
                    _pos_set(pos_store, str(n), x * POS_SCALE, y * POS_SCALE)

            node_classes = {n: "" for n in start_nodes}
            elements = render_graph(G_start, start_nodes, node_classes)

            transition = {
                "network_kind": network_kind,
                "start_year": start_year,
                "target_year": year,
                "min_value": min_value,
                "sim_threshold": sim_threshold,
                "node_scaling": node_scaling,
                "edge_scaling": edge_scaling,
                "community": community,
                "persistent": sorted(persistent),
                "deaths": sorted(deaths),
                "births": sorted(births),
                "stage": "fade",
                "stage_step": 0,
                "move_step": 0,
                "move_start_pos": None,
                "move_target_pos": None,
            }

            meta = [
                html.Div(f"Transition: {start_year} → {year} (staged)"),
                html.Div(
                    f"Stage: fade deaths  |  persistent={len(persistent)}  births={len(births)}  deaths={len(deaths)}"
                ),
            ]
            cam_nodes = active_nodes if "active_nodes" in locals() else start_nodes
            return _ret(
                elements, meta, False, start_year, pos_store, transition, set(cam_nodes)
            )

        # -----------------------------
        # Interval tick: advance staged transition
        # -----------------------------
        stage = transition.get("stage", "fade")
        stage_step = int(transition.get("stage_step", 0))

        network_kind = transition["network_kind"]
        start_year = int(transition["start_year"])
        target_year = int(transition["target_year"])
        min_value = int(transition["min_value"])
        sim_threshold = float(transition["sim_threshold"])
        node_scaling = transition["node_scaling"]
        edge_scaling = transition["edge_scaling"]
        community = transition["community"]

        G_start = get_graph(
            network_kind, start_year, min_value=min_value, sim_threshold=sim_threshold
        )
        G_target = get_graph(
            network_kind, target_year, min_value=min_value, sim_threshold=sim_threshold
        )

        start_nodes = {str(n) for n in G_start.nodes()}
        target_nodes = {str(n) for n in G_target.nodes()}

        persistent = set(transition.get("persistent", []))
        deaths = set(transition.get("deaths", []))
        births = set(transition.get("births", []))

        # 1) Fade deaths (persisting nodes stay put; no movement)
        if stage == "fade":
            stage_step += 1
            transition["stage_step"] = stage_step

            node_classes = {}
            for n in start_nodes:
                if n in deaths:
                    node_classes[n] = "inactive"
                else:
                    node_classes[n] = ""

            # fade edges incident to deaths at the end of fade stage
            edge_classes = {}
            for u, v in G_start.edges():
                su, sv = str(u), str(v)
                if su in deaths or sv in deaths:
                    edge_classes[_edge_key(su, sv)] = "inactive-edge"

            elements = render_graph(
                G_start, start_nodes, node_classes, edge_classes=edge_classes
            )

            if stage_step >= FADE_STEPS:
                # Lock community colors for the target graph once deaths are gone (prevents flicker).
                if "locked_node_colors" not in transition:
                    cm = (
                        _community_color_map(G_target, community)
                        if G_target.number_of_nodes()
                        else {}
                    )
                    locked: dict[str, str] = {}
                    for node, data in G_target.nodes(data=True):
                        nid = str(node)
                        comm = data.get(community, 0)
                        locked[nid] = cm.get(comm, "rgb(160,160,160)")
                    transition["locked_node_colors"] = locked
                transition["stage"] = "birth"
                transition["stage_step"] = 0
            meta = [
                html.Div(f"Transition: {start_year} → {target_year}"),
                html.Div(
                    f"Stage: fade deaths ({min(stage_step, FADE_STEPS)}/{FADE_STEPS})"
                ),
            ]
            cam_nodes = active_nodes if "active_nodes" in locals() else start_nodes
            return _ret(
                elements, meta, False, start_year, pos_store, transition, set(cam_nodes)
            )

        # 2) Births fade in near neighbors (still no movement)
        if stage == "birth":
            stage_step += 1
            transition["stage_step"] = stage_step

            active_nodes = set(persistent) | set(births)

            # ensure births have initial positions near their neighbors
            for b in births:
                if _pos_get(pos_store, b) is None:
                    c = _neighbors_centroid(G_target, b, pos_store)
                    if c is None:
                        c = (0.0, 0.0)
                    x0 = c[0] + random.uniform(-20, 20)
                    y0 = c[1] + random.uniform(-20, 20)
                    _pos_set(pos_store, b, x0, y0)

            node_classes = {n: "" for n in persistent}
            for b in births:
                node_classes[b] = "incoming" if stage_step < BIRTH_STEPS else ""

            # incoming edges (optional): edges incident to births start faint
            edge_classes = {}
            for u, v in G_target.edges():
                su, sv = str(u), str(v)
                if su in births or sv in births:
                    if stage_step < BIRTH_STEPS:
                        edge_classes[_edge_key(su, sv)] = "incoming-edge"

            elements = render_graph(
                G_target,
                active_nodes,
                node_classes,
                edge_classes=edge_classes,
                locked_node_colors=transition.get("locked_node_colors"),
            )

            if stage_step >= BIRTH_STEPS:
                # snapshot start positions for move stage
                move_start_pos = {
                    n: list(_pos_get(pos_store, n) or (0.0, 0.0)) for n in active_nodes
                }

                # compute target layout for active nodes, initialised from current positions
                init_pos = {
                    n: (_pos_get(pos_store, n) or (0.0, 0.0)) for n in active_nodes
                }
                G_active = G_target.subgraph(list(active_nodes)).copy()
                layout = spring_layout_cyto(G_active, init_pos_cyto=init_pos)
                move_target_pos = {
                    n: [layout.get(n, init_pos[n])[0], layout.get(n, init_pos[n])[1]]
                    for n in active_nodes
                }

                # align so the whole map doesn't sweep/rotate
                move_target_pos = align_target_to_start(
                    move_start_pos,
                    move_target_pos,
                    list(persistent) if persistent else list(active_nodes),
                )

                transition["stage"] = "move"
                transition["stage_step"] = 0
                transition["move_step"] = 0
                transition["move_start_pos"] = move_start_pos
                transition["move_target_pos"] = move_target_pos

            meta = [
                html.Div(f"Transition: {start_year} → {target_year}"),
                html.Div(
                    f"Stage: births fade in ({min(stage_step, BIRTH_STEPS)}/{BIRTH_STEPS})"
                ),
            ]
            cam_nodes = active_nodes if "active_nodes" in locals() else start_nodes
            return _ret(
                elements, meta, False, start_year, pos_store, transition, set(cam_nodes)
            )

        # 3) Gentle settle (NOW allow all nodes to move)
        if stage == "move":
            move_step = int(transition.get("move_step", 0)) + 1
            transition["move_step"] = move_step

            active_nodes = set(persistent) | set(births)

            move_start_pos = transition.get("move_start_pos") or {}
            move_target_pos = transition.get("move_target_pos") or {}

            t = _smoothstep(move_step / max(1, MOVE_STEPS))

            for n in active_nodes:
                sp = move_start_pos.get(n, [0.0, 0.0])
                tp = move_target_pos.get(n, sp)

                dx = (tp[0] - sp[0]) * t
                dy = (tp[1] - sp[1]) * t

                dist = (dx * dx + dy * dy) ** 0.5
                if dist > MAX_MOVE_PER_TRANSITION:
                    k = MAX_MOVE_PER_TRANSITION / (dist + 1e-9)
                    dx *= k
                    dy *= k

                _pos_set(pos_store, n, sp[0] + dx, sp[1] + dy)

            node_classes = {n: "" for n in active_nodes}
            elements = render_graph(
                G_target,
                active_nodes,
                node_classes,
                locked_node_colors=transition.get("locked_node_colors"),
            )

            done = move_step >= MOVE_STEPS
            if done:
                for n in active_nodes:
                    tp = move_target_pos.get(n)
                    if tp:
                        _pos_set(pos_store, n, tp[0], tp[1])
                meta = [
                    html.Div(f"Year: {target_year} (settled)"),
                    html.Div(
                        f"Nodes: {len(active_nodes)}  |  Edges: {G_target.subgraph(list(active_nodes)).number_of_edges()}"
                    ),
                ]
                return _ret(
                    elements, meta, True, target_year, pos_store, None, active_nodes
                )

            meta = [
                html.Div(f"Transition: {start_year} → {target_year}"),
                html.Div(
                    f"Stage: gentle settle ({min(move_step, MOVE_STEPS)}/{MOVE_STEPS})"
                ),
            ]
            cam_nodes = active_nodes if "active_nodes" in locals() else start_nodes
            return _ret(
                elements, meta, False, start_year, pos_store, transition, set(cam_nodes)
            )

        # fallback
        return _ret(
            [], [html.Div("No graph")], True, target_year, pos_store, transition, set()
        )

    @app.callback(
        Output("temp-details", "children"),
        Input("temp-cyto", "tapNodeData"),
        Input("temp-cyto", "tapEdgeData"),
        State("temp-network", "value"),
        prevent_initial_call=False,
    )
    def show_temporal_selection(node_data, edge_data, network_kind):
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

        # Edge
        src = edge_data.get("source")
        tgt = edge_data.get("target")
        raw_w = edge_data.get("raw_weight")
        return html.Div(
            [
                html.Div([html.B("Edge:"), f" {src} — {tgt}"]),
                html.Div([html.B("raw_weight:"), f" {raw_w}"]),
                html.Div(
                    "Next upgrade: show supporting publications where both endpoints occur together."
                ),
            ]
        )

    @app.callback(
        Output("temp-bipartite-headers", "style"),
        Input("temp-network", "value"),
    )
    def toggle_temp_bip_headers(network_kind):
        if str(network_kind).startswith("concept_method"):
            return {
                "display": "flex",
                "justifyContent": "space-between",
                "padding": "0 18px 6px 18px",
                "fontWeight": 600,
                "color": "#444",
            }
        return {"display": "none"}
