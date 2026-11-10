"""Versioned HAI 20.07 observed-tag graph, with evidence for every semantic edge."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch

from src.config import PROJECT_ROOT
from src.preprocess_hai_mixed import sha256


COLORS = {"P1": "#4c9bd6", "P2": "#e69f42", "P3": "#58ad82", "P4": "#ab83c6"}
ROLES = ["measurement", "command", "setpoint", "state"]


def node_role(name, binary_columns):
    if name in binary_columns:
        return "state"
    if name == "P1_B4005" or (name.startswith("P1_") and name.endswith("D")) or name in ("P3_LCP01D", "P3_LCV01D"):
        return "command"
    if name.startswith("P1_B") or name in ("P2_SD01", "P3_LH", "P3_LL", "P4_LD") or name.endswith(("_PS", "_LD")):
        return "setpoint"
    return "measurement"


def build_graph(schema):
    order = schema["feature_order"]
    if len(order) != 59 or len(set(order)) != 59:
        raise ValueError("Expected 59 unique HAI 20.07 channels")
    nodes = [{"id": i, "name": name, "process": name[:2],
              "role": node_role(name, schema["binary_columns"])} for i, name in enumerate(order)]
    edges = {}

    def link(a, b, relation, evidence, reference):
        if a not in order or b not in order or a == b:
            raise ValueError(f"Invalid semantic edge: {a}, {b}")
        pair = tuple(sorted((a, b)))
        item = {"relation": relation, "evidence": evidence, "reference": reference}
        edges.setdefault(pair, []).append(item)

    def function(a, b, relation, reference):
        link(a, b, relation, "documented_functional_projection", reference)

    def prior(a, b, relation, reference):
        link(a, b, relation, "engineering_prior_not_direct_wiring", reference)

    for valve in ("FCV01", "FCV02", "FCV03", "LCV01", "PCV01", "PCV02"):
        link(f"P1_{valve}D", f"P1_{valve}Z", "command_and_position_feedback",
             "documented_tag_pair", "HAI DATA POINTS, printed pp.12-13")
    for flow in ("FT01", "FT02", "FT03"):
        link(f"P1_{flow}", f"P1_{flow}Z", "measurement_and_converted_value",
             "documented_tag_pair", "HAI DATA POINTS, printed p.12")

    # PID, plant and controller nodes are unobserved: project their functional paths onto tags.
    for command in ("P1_PCV01D", "P1_PCV02D"):
        function("P1_B2016", command, "pressure_setpoint_to_control", "Fig.4, printed p.7")
        function("P1_PIT01", command, "pressure_feedback_control_loop", "Fig.4, printed p.7")
    function("P1_B3004", "P1_LCV01D", "level_setpoint_to_control", "Fig.5, printed p.7")
    function("P1_LIT01", "P1_LCV01D", "level_feedback_control_loop", "Fig.5, printed p.7")
    function("P1_FCV03D", "P1_LCV01D", "outflow_feedforward", "Fig.5, printed p.7")
    function("P1_B3005", "P1_FCV03D", "flow_setpoint_to_control", "Fig.6, printed p.7")
    function("P1_FT03", "P1_FCV03D", "flow_feedback_control_loop", "Fig.6, printed p.7")
    for command in ("P1_FCV01D", "P1_FCV02D"):
        function("P1_B4022", command, "temperature_setpoint_to_control", "Fig.7, printed p.8")
        function("P1_TIT01", command, "temperature_feedback_control_loop", "Fig.7, printed p.8")
        function("P1_FT02", command, "cascade_flow_compensation", "Fig.7, printed p.8")

    # Tag semantics / plant physics priors: explicit assumptions, not exported DCS wiring.
    for a, b, relation in (
        ("P1_B2004", "P1_PIT01", "outlet_pressure_setpoint_and_measurement"),
        ("P1_B4002", "P1_TIT01", "outlet_temperature_setpoint_and_measurement"),
        ("P1_B4005", "P1_TIT01", "temperature_PID_output_context"),
        ("P1_B400B", "P1_FT02Z", "heating_tank_outflow_demand_context"),
        ("P1_FT01Z", "P1_LIT01", "tank_inflow_level_coupling"),
        ("P1_FT03Z", "P1_LIT01", "tank_outflow_level_coupling"),
        ("P1_TIT02", "P1_TIT01", "heating_water_heat_transfer"),
        ("P1_PIT02", "P1_FT02", "heating_water_pressure_flow_coupling"),
    ):
        prior(a, b, relation, "HAI DATA POINTS pp.12-13 and process description pp.4-5; interpreted")

    function("P2_SD01", "P2_SIT01", "speed_setpoint_feedback_loop", "Fig.11, printed p.10; SD01=AutoSD")
    for state in ("P2_Auto", "P2_On"):
        prior(state, "P2_SIT01", "controller_enable_speed_context", "Fig.10 and DATA POINTS pp.13-14; interpreted")
    function("P2_SIT01", "P2_Emgy", "overspeed_trip", "P2-TRIP text, printed p.10")
    function("P2_TripEx", "P2_Emgy", "trip_reset", "P2-TRIP text, printed p.10")
    for vibration in ("P2_VXT02", "P2_VXT03", "P2_VYT02", "P2_VYT03"):
        function(vibration, "P2_Emgy", "vibration_trip", "P2-TRIP, p.10; HAI20.07 aliases, p.14")
        prior("P2_SIT01", vibration, "shared_rotating_shaft_context", "Fig.10, printed p.10; interpreted")

    for command in ("P3_LCP01D", "P3_LCV01D"):
        function("P3_LT01", command, "reservoir_level_feedback_via_HTM", "Figs.12-13, printed p.11")
        for setpoint in ("P3_LH", "P3_LL"):
            function(setpoint, command, "level_threshold_via_HTM", "Fig.13 p.11; LH/LL aliases from DATA POINTS")
        prior(command, "P4_HT_PO", "water_actuation_hydropower_coupling", "Process description printed p.5; interpreted tag-level projection")

    function("P1_PIT01", "P4_ST_PT01", "boiler_to_simulated_steam_pressure", "Process description pp.4-5, DATA POINTS p.15")
    function("P1_TIT01", "P4_ST_TT01", "boiler_to_simulated_steam_temperature", "Process description pp.4-5, DATA POINTS p.15")
    for family in ("ST", "HT"):
        for demand in ("PS", "LD"):
            prior(f"P4_{family}_{demand}", f"P4_{family}_PO", "power_demand_generation_context", "HIL description p.5, DATA POINTS pp.14-15; interpreted")
        prior(f"P4_{family}_PO", f"P4_{family}_FD", "generation_frequency_context", "HIL description p.5; interpreted")
        prior("P4_LD", f"P4_{family}_LD", "grid_and_component_load_context", "HIL description p.5, DATA POINTS p.15; interpreted")
    for source in ("P4_ST_PT01", "P4_ST_TT01"):
        prior(source, "P4_ST_PO", "steam_state_generation_context", "Process description pp.4-5; interpreted")
    prior("P4_ST_FD", "P2_SIT01", "HIL_frequency_speed_synchronization", "Fig.11 and HIL text, printed pp.5,10; interpreted")
    prior("P4_HT_FD", "P4_ST_FD", "shared_simulated_grid", "HIL description printed p.5; interpreted")
    prior("P3_LT01", "P4_HT_PO", "reservoir_state_hydropower", "Fig.12 / HIL description printed pp.5,11; interpreted")

    semantic = [{"source": a, "target": b, "evidence": records}
                for (a, b), records in sorted(edges.items())]
    index = {node["name"]: node["id"] for node in nodes}
    pairs = {(i, i) for i in range(len(nodes))}
    for edge in semantic:
        a, b = index[edge["source"]], index[edge["target"]]
        pairs.update(((a, b), (b, a)))
    edge_index = torch.tensor(sorted(pairs), dtype=torch.long).t().contiguous()
    static = np.zeros((len(nodes), 8), dtype=np.float32)
    for node in nodes:
        static[node["id"], int(node["process"][1])-1] = 1.
        static[node["id"], 4+ROLES.index(node["role"])] = 1.
    return {
        "version": "hai20_07_tag_graph_v1", "node_order": order, "nodes": nodes,
        "edges": semantic, "undirected_functional_graph": True,
        "note": "Observed-tag functional graph, not a network packet topology or proven causal graph. Undirected GCN symmetrization with one explicit self-loop per node.",
        "isolated_tag_policy": "P2_24Vdc and P2_VT01e retain only self-loops: no asserted specific edge; VT01e alias is not fully documented.",
    }, edge_index, torch.from_numpy(static)


def plot_graph(graph, output):
    g = nx.Graph()
    for node in graph["nodes"]:
        g.add_node(node["name"], **node)
    for edge in graph["edges"]:
        g.add_edge(edge["source"], edge["target"],
                   interpreted=all(e["evidence"].startswith("engineering") for e in edge["evidence"]))
    positions = {}
    for column, process in enumerate(COLORS):
        names = [n["name"] for n in graph["nodes"] if n["process"] == process]
        for name, y in zip(names, np.linspace(1, 30, len(names))):
            positions[name] = (column*4, y)
    fig, ax = plt.subplots(figsize=(20, 16))
    for interpreted, style in ((False, "solid"), (True, "dashed")):
        selected = [(a, b) for a, b, data in g.edges(data=True) if data["interpreted"] == interpreted]
        nx.draw_networkx_edges(g, positions, edgelist=selected, ax=ax, alpha=.25,
                               edge_color="#8b8b8b" if interpreted else "#335b87", style=style)
    nx.draw_networkx_nodes(g, positions, ax=ax, node_size=180,
                           node_color=[COLORS[g.nodes[name]["process"]] for name in g])
    nx.draw_networkx_labels(g, {name: (x+.14, y) for name, (x, y) in positions.items()},
                            ax=ax, font_size=8, horizontalalignment="left")
    for column, (process, color) in enumerate(COLORS.items()):
        ax.text(column*4, 31.2, process, fontsize=16, weight="bold", color=color)
    ax.set_title("HAI 20.07 observed-tag graph: 59 nodes\nSolid: documented functional/tag paths; dashed: interpreted engineering priors; self-loops omitted", fontsize=14)
    ax.set_xlim(-.5, 14.5)
    ax.set_ylim(0, 32)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "tag_graph.png", dpi=150)
    plt.close(fig)
    # Separate panels expose internal edges that overlap in the column overview.
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    for ax, (process, color) in zip(axes.flat, COLORS.items()):
        names = [name for name in g if name.startswith(process)]
        sub = g.subgraph(names)
        layout = nx.circular_layout(sub)
        for interpreted, style in ((False, 'solid'), (True, 'dashed')):
            links = [(a,b) for a,b,d in sub.edges(data=True) if d['interpreted'] == interpreted]
            nx.draw_networkx_edges(sub, layout, edgelist=links, ax=ax, style=style,
                                   edge_color='#888888' if interpreted else '#335b87', alpha=.65)
        nx.draw_networkx_nodes(sub, layout, ax=ax, node_color=color, node_size=160)
        label_pos = {name: point*1.12 for name, point in layout.items()}
        nx.draw_networkx_labels(sub, label_pos, ax=ax, font_size=8)
        ax.set_title(f'{process}: {len(names)} tags; {sub.number_of_edges()} internal links')
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.3, 1.3); ax.axis('off')
    fig.suptitle('HAI internal functional graphs: solid documented, dashed engineering prior', fontsize=18)
    fig.tight_layout()
    fig.savefig(output / 'process_detail.png', dpi=150)
    plt.close(fig)
    overview = nx.Graph()
    overview.add_nodes_from(COLORS)
    for a, b in g.edges:
        pa, pb = a[:2], b[:2]
        if pa != pb:
            overview.add_edge(pa, pb, weight=overview.get_edge_data(pa, pb, {}).get("weight", 0)+1)
    pos = {"P1": (-1, 1), "P2": (1, 1), "P3": (-1, -1), "P4": (1, -1)}
    fig, ax = plt.subplots(figsize=(9, 7))
    labels = {p: f"{p}\n{sum(n['process']==p for n in graph['nodes'])} observed tags" for p in COLORS}
    nx.draw(overview, pos, ax=ax, labels=labels, node_color=list(COLORS.values()),
            node_size=4500, font_size=11, edge_color="#7c8793", width=2)
    nx.draw_networkx_edge_labels(overview, pos, ax=ax,
                                edge_labels={(a,b): f"{d['weight']} tag links" for a,b,d in overview.edges(data=True)})
    ax.set_title("Process overview: P1 boiler, P2 turbine, P3 water, P4 HIL\nFunctional/engineering prior graph; not physical wiring")
    fig.tight_layout()
    fig.savefig(output / "process_overview.png", dpi=150)
    plt.close(fig)


def main():
    schema = json.loads((PROJECT_ROOT / "configs/hai_20_07_schema.json").read_text())
    graph, edge_index, static = build_graph(schema)
    graph["source_pdf_sha256"] = sha256(PROJECT_ROOT / "docs/references/hai_dataset_technical_details.pdf")
    output = PROJECT_ROOT / "artifacts/hai/graphs" / graph["version"]
    if output.exists():
        raise FileExistsError("Graph version already exists; do not silently replace it")
    output.mkdir(parents=True)
    (output / "graph.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")
    graph_hash = sha256(output / "graph.json")
    torch.save({"edge_index": edge_index, "node_static": static,
                "node_order": graph["node_order"], "graph_sha256": graph_hash}, output / "graph.pt")
    pd.DataFrame(graph["nodes"]).to_csv(output / "nodes.csv", index=False)
    records = [{"source": edge["source"], "target": edge["target"], **evidence}
               for edge in graph["edges"] for evidence in edge["evidence"]]
    pd.DataFrame(records).to_csv(output / "edges_with_evidence.csv", index=False)
    plot_graph(graph, output)
    print(f"Nodes={len(graph['nodes'])}; semantic undirected pairs={len(graph['edges'])}; "
          f"GCN edges including reverse edges and self-loops={edge_index.shape[1]}")
    print("Saved:", output)


if __name__ == "__main__":
    main()
