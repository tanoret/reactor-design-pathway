import json
import os
import re
from typing import Dict, Any, List, Tuple, Optional

import streamlit as st

# Optional libs for graph rendering (PyVis fallback)
try:
    import networkx as nx
    from pyvis.network import Network
    HAS_GRAPH = True
except Exception:
    HAS_GRAPH = False

# Optional interactive graph (click-to-select)
try:
    from streamlit_agraph import agraph, Node, Edge, Config  # pip install streamlit-agraph
    HAS_AGRAPH = True
except Exception:
    HAS_AGRAPH = False

# Dataframes for the Data Browser
try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

st.set_page_config(page_title="Design and Engineering for Licensing Process — Graph Viewer", layout="wide")
DEFAULT_JSON_PATH = "data/nrc_licensing_process_data.json"


# -------------------------
# Utilities
# -------------------------
def wrap_text(s: str, width: int = 28, max_lines: int = 3) -> str:
    """Insert newlines to keep node labels tidy."""
    if not s:
        return ""
    parts, line = [], []
    count = 0
    for word in s.split():
        if count + len(word) + (1 if line else 0) <= width:
            line.append(word)
            count += len(word) + (1 if line[:-1] else 0)
        else:
            parts.append(" ".join(line))
            line = [word]; count = len(word)
            if max_lines and len(parts) >= max_lines - 1:
                break
    if line and (not max_lines or len(parts) < max_lines):
        parts.append(" ".join(line))
    # truncate if exceeded
    if len(parts) > max_lines:
        parts = parts[:max_lines]
    out = "\n".join(parts)
    if s != " ".join((" ".join(p.split()) for p in parts)):
        out += " …"
    return out


def compute_levels(tasks: List[Dict[str, Any]]) -> Dict[str, int]:
    """Rough depth (longest distance from any root) for hierarchical layout."""
    # Build graph
    g = nx.DiGraph()
    for t in tasks:
        tid = str(t.get("id"))
        g.add_node(tid)
    for t in tasks:
        tid = str(t.get("id"))
        for d in (t.get("deps") or []):
            g.add_edge(str(d), tid)
    # If cycles, fall back gracefully
    try:
        order = list(nx.topological_sort(g))
    except Exception:
        return {}

    depth = {n: 0 for n in g.nodes}
    for n in order:
        for _, v in g.out_edges(n):
            depth[v] = max(depth[v], depth[n] + 1)
    return depth


# -------------------------
# Data loading & filtering
# -------------------------
@st.cache_data
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_json_bytes(b: bytes) -> Dict[str, Any]:
    return json.loads(b.decode("utf-8"))

def infer_filters(data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    phases = sorted({t.get("phase") for t in data.get("tasks", []) if t.get("phase")})
    chapters = sorted({t.get("srpChapter") for t in data.get("tasks", []) if t.get("srpChapter")})
    return phases, chapters

def text_blob(task: Dict[str, Any]) -> str:
    fields: List[str] = [
        str(task.get("id","")), str(task.get("name","")), str(task.get("phase","") or ""),
        str(task.get("srpChapter") or ""), str(task.get("description") or "")
    ]
    for k in ("whatIsInvolved","mustAccomplish","tags","deliverables","acceptance"):
        vals = task.get(k) or []
        if isinstance(vals, list):
            fields.extend([str(v) for v in vals])
        else:
            fields.append(str(vals))
    for r in task.get("requirements") or []:
        fields.append(str(r.get("type","")))
        fields.append(str(r.get("cite","")))
        fields.append(str(r.get("topic") or ""))
    for s in task.get("subtasks") or []:
        fields.append(str(s.get("title","")))
        fields.extend([str(v) for v in (s.get("details") or [])])
        for tool in s.get("tools") or []:
            fields.append(str(tool.get("name","")))
            fields.append(str(tool.get("type","")))
            fields.append(str(tool.get("purpose","")))
    return " ".join(fields).lower()

def filter_tasks(data: Dict[str, Any], query: str, phases_sel: List[str], chapters_sel: List[str]) -> List[Dict[str, Any]]:
    tasks = data.get("tasks", [])
    q = (query or "").strip().lower()
    res = []
    for t in tasks:
        if phases_sel and t.get("phase") not in phases_sel:
            continue
        if chapters_sel and (t.get("srpChapter") or "") not in chapters_sel:
            continue
        if q and q not in text_blob(t):
            continue
        res.append(t)
    return res

def color_for_phase(phase: str) -> str:
    palette = {
        "Orientation": "#E6F3FF",
        "Site & NEPA": "#E7F8ED",
        "Engineering Process": "#FDEDD7",
        "SRP Chapters": "#F3E8FF",
        "Security & Cyber": "#FDE2E1",
        "EP": "#FFF5CC",
        "ITAAC & Tests": "#E5FBFF",
        "Operations Programs": "#E7F0FF",
        "Cross-Cutting": "#EFEFEF",
        "Alternatives": "#F5E8D8",
        "Project Controls": "#E8FFE8",
        "Traceability": "#F0F0FF",
        "Verification": "#E8F7FF",
        "Readiness": "#FFE8F1",
        "Tailoring": "#EAFBEA",
        None: "#FFFFFF",
        "": "#FFFFFF",
    }
    return palette.get(phase, "#FFFFFF")


# -------------------------
# Graph rendering (PyVis) — readable defaults
# -------------------------
def build_graph_html(tasks: List[Dict[str, Any]], hierarchical: bool = True, physics: bool = False) -> str:
    if not HAS_GRAPH:
        return "<div style='padding:16px;font-family:system-ui'>Install dependencies to render graph: <code>pip install networkx pyvis</code></div>"

    import networkx as nx
    from pyvis.network import Network

    levels = compute_levels(tasks) if hierarchical else {}

    G = nx.DiGraph()
    for t in tasks:
        nid = str(t.get("id"))
        name = t.get("name","")
        label = f"{t.get('id','')}\n{wrap_text(name)}".strip()
        title = f"<b>{t.get('id','')}</b> — {name}"
        chapter = t.get("srpChapter") or ""
        phase = t.get("phase") or ""
        meta = []
        if phase:
            meta.append("<span style='border:1px solid #c7d2fe;border-radius:999px;padding:2px 6px;background:#eef2ff;color:#3730a3;'>{}</span>".format(phase))
        if chapter:
            meta.append("<span style='border:1px solid #c7d2fe;border-radius:999px;padding:2px 6px;background:#eef2ff;color:#3730a3;'>{}</span>".format(chapter))
        desc = (t.get("description") or "").replace("<", "&lt;").replace(">", "&gt;")

        tool_names = []
        for s in (t.get("subtasks") or []):
            for tool in (s.get("tools") or [])[:1]:
                tool_names.append(tool.get("name",""))
        preview = ""
        if tool_names:
            preview = "<div style='margin-top:6px;font-size:11px;color:#475569;'>Tools: " + ", ".join(tool_names[:5]) + "</div>"

        title_html = "{}<br/>{}<br/><div style='margin-top:6px;color:#374151;font-size:12px'>{}</div>{}".format(
            title, " ".join(meta), desc, preview
        )

        attrs = dict(label=label, title=title_html, color=color_for_phase(phase), shape="box")
        if hierarchical and levels:
            attrs["level"] = int(levels.get(nid, 0))
        G.add_node(nid, **attrs)

    for t in tasks:
        for d in t.get("deps") or []:
            if str(d) in G.nodes and str(t.get("id")) in G.nodes:
                G.add_edge(str(d), str(t.get("id")))

    net = Network(
        height="700px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#222",
        cdn_resources="in_line",  # avoid CDN stalls
    )

    if hierarchical:
        options = """
        {
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "LR",
              "sortMethod": "directed",
              "nodeSpacing": 170,
              "treeSpacing": 220,
              "levelSeparation": 160
            }
          },
          "physics": { "enabled": %s },
          "interaction": { "hover": true, "tooltipDelay": 100, "navigationButtons": false, "keyboard": { "enabled": true } },
          "edges": { "arrows": { "to": { "enabled": true } } },
          "nodes": { "shape": "box", "font": { "size": 14 } },
          "configure": { "enabled": false }
        }
        """ % (str(physics).lower())
    else:
        options = """
        {
          "layout": { "improvedLayout": true },
          "physics": {
            "enabled": %s,
            "solver": "barnesHut",
            "stabilization": { "enabled": true, "fit": true },
            "barnesHut": { "gravitationalConstant": -30000, "centralGravity": 0.2,
                           "springLength": 160, "springConstant": 0.05, "damping": 0.8 }
          },
          "interaction": { "hover": true, "tooltipDelay": 100, "navigationButtons": true, "keyboard": { "enabled": true } },
          "edges": { "arrows": { "to": { "enabled": true } } },
          "nodes": { "shape": "box", "font": { "size": 14 } },
          "configure": { "enabled": false }
        }
        """ % (str(physics).lower())

    net.set_options(options)
    net.from_nx(G)
    return net.generate_html()


# -------------------------
# Graph rendering (Agraph) with click feedback
# -------------------------
def render_graph_interactive(
    tasks: List[Dict[str, Any]],
    use_agraph: bool = True,
    hierarchical: bool = True,
    physics: bool = False
) -> Optional[str]:
    """Return clicked node id if using agraph; PyVis fallback returns None."""
    if not tasks:
        return None

    if HAS_AGRAPH and use_agraph:
        nodes: List[Node] = []
        edges: List[Edge] = []

        levels = compute_levels(tasks) if hierarchical else {}

        for t in tasks:
            nid = str(t.get("id"))
            name = t.get("name","")
            label = f"{t.get('id','')} — {wrap_text(name)}"
            phase = t.get("phase") or ""
            desc = t.get("description") or ""
            node_kwargs = dict(
                id=nid,
                label=label,
                title=f"<b>{nid}</b><br/>{desc}",
                size=18,
                shape="box",
                color=color_for_phase(phase),
            )
            # agraph Node accepts 'level' and 'font' in recent versions; ignore if unsupported
            if hierarchical and levels:
                node_kwargs["level"] = int(levels.get(nid, 0))
            nodes.append(Node(**node_kwargs))

        for t in tasks:
            for d in t.get("deps") or []:
                edges.append(Edge(source=str(d), target=str(t.get("id"))))

        cfg = Config(
            width=900, height=700, directed=True,
            physics=physics,
            hierarchical=hierarchical,
            nodeHighlightBehavior=True,
        )

        ret = agraph(nodes=nodes, edges=edges, config=cfg)
        # Normalize return value across versions
        if isinstance(ret, dict):
            cand = ret.get("selected_node") or ret.get("clicked_node")
            if not cand:
                arr = ret.get("selected_nodes") or []
                cand = arr[0] if arr else None
            return str(cand) if cand else None
        if isinstance(ret, (list, tuple)) and ret:
            return str(ret[0])
        if isinstance(ret, str) and ret:
            return ret
        return None

    # Fallback (no click feedback)
    html = build_graph_html(tasks, hierarchical=hierarchical, physics=physics)
    st.components.v1.html(html, height=740, scrolling=True)
    return None


# -------------------------
# Details panel
# -------------------------
def show_list(items: List[str]):
    if items:
        for x in items:
            st.markdown(f"- {x}")
    else:
        st.write("—")

def details_block(task: Dict[str, Any]):
    st.subheader(f"{task.get('id')} — {task.get('name')}")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Phase:** " + str(task.get("phase") or "—"))
    with cols[1]:
        st.markdown("**SRP Chapter:** " + str(task.get("srpChapter") or "—"))
    with cols[2]:
        st.markdown("**Tags:** " + (", ".join(task.get("tags") or []) if task.get("tags") else "—"))

    st.markdown("### Description")
    st.write(task.get("description") or "—")

    st.markdown("### What is involved")
    show_list(task.get("whatIsInvolved") or [])

    st.markdown("### Must accomplish")
    show_list(task.get("mustAccomplish") or [])

    st.markdown("### Requirements (CFR / RG / NUREG / Standards)")
    reqs = task.get("requirements") or []
    if reqs:
        for r in reqs:
            t = r.get("type", "")
            c = r.get("cite", "")
            topic = r.get("topic") or ""
            st.markdown(f"- **[{t}]** {c}{' — ' + topic if topic else ''}")
    else:
        st.write("—")

    st.markdown("### Deliverables")
    show_list(task.get("deliverables") or [])

    st.markdown("### Acceptance / Completion")
    show_list(task.get("acceptance") or [])

    subs = task.get("subtasks") or []
    st.markdown("### Engineering Subtasks")
    if subs:
        for i, s in enumerate(subs, start=1):
            with st.expander(f"{i}. {s.get('title') or 'Subtask'}", expanded=False):
                details = s.get("details") or []
                if details:
                    st.markdown("**Details**")
                    for d in details:
                        st.markdown(f"- {d}")
                tools = s.get("tools") or []
                st.markdown("**Recommended computational tools**")
                if tools:
                    for tool in tools:
                        name = tool.get("name","")
                        typ = tool.get("type","")
                        purpose = tool.get("purpose","")
                        st.markdown(f"- **{name}** *({typ})* — {purpose}")
                else:
                    st.write("—")
    else:
        st.write("—")


# -------------------------
# Data Browser helpers
# -------------------------
def build_dataframes(data: Dict[str, Any]):
    tasks = data.get("tasks", [])
    rows_tasks, rows_reqs, rows_tools, rows_subtasks = [], [], [], []
    for t in tasks:
        tid = str(t.get("id",""))
        name = t.get("name","")
        phase = t.get("phase")
        srp = t.get("srpChapter")
        tags = t.get("tags") or []
        reqs = t.get("requirements") or []
        subs = t.get("subtasks") or []

        for r in reqs:
            rows_reqs.append({
                "task_id": tid,
                "task_name": name,
                "phase": phase,
                "srpChapter": srp,
                "req_type": r.get("type",""),
                "req_cite": r.get("cite",""),
                "req_topic": r.get("topic","")
            })

        tool_names, tool_types = [], set()
        for s in subs:
            sid = s.get("id","")
            stitle = s.get("title","")
            rows_subtasks.append({
                "task_id": tid, "task_name": name, "phase": phase, "srpChapter": srp,
                "subtask_id": sid, "subtask_title": stitle
            })
            for tool in (s.get("tools") or []):
                rows_tools.append({
                    "task_id": tid, "task_name": name, "phase": phase, "srpChapter": srp,
                    "subtask_id": sid, "subtask_title": stitle,
                    "tool_name": tool.get("name",""),
                    "tool_type": tool.get("type",""),
                    "tool_purpose": tool.get("purpose",""),
                })
                tool_names.append(tool.get("name",""))
                tool_types.add(tool.get("type",""))

        req_types = sorted({r.get("type","") for r in reqs if r})
        req_cites = "; ".join([
            (r.get("cite","") + (f" — {r.get('topic')}" if r.get("topic") else ""))
            for r in reqs
        ])

        rows_tasks.append({
            "id": tid,
            "name": name,
            "phase": phase,
            "srpChapter": srp,
            "tags": ", ".join(tags),
            "description": t.get("description",""),
            "whatIsInvolved": "\n".join(t.get("whatIsInvolved") or []),
            "mustAccomplish": "\n".join(t.get("mustAccomplish") or []),
            "deliverables": "\n".join(t.get("deliverables") or []),
            "acceptance": "\n".join(t.get("acceptance") or []),
            "requirements_count": len(reqs),
            "requirements_types": ", ".join(req_types),
            "requirements_cites": req_cites,
            "subtasks_count": len(subs),
            "tools_count": len(tool_names),
            "tool_names": ", ".join(tool_names),
            "tool_types": ", ".join(sorted(tool_types)),
        })

    df_tasks = pd.DataFrame(rows_tasks) if HAS_PANDAS else None
    df_reqs = pd.DataFrame(rows_reqs) if HAS_PANDAS else None
    df_tools = pd.DataFrame(rows_tools) if HAS_PANDAS else None
    df_subtasks = pd.DataFrame(rows_subtasks) if HAS_PANDAS else None
    return df_tasks, df_reqs, df_tools, df_subtasks

def df_download_button(df: pd.DataFrame, label: str, filename: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv, file_name=filename, mime="text/csv")


# -------------------------
# App
# -------------------------
def main():
    st.title("Reactor Design and Engineering Process")
    st.caption("Readable graph layout + reliable click-to-select (no locking).")

    # Sidebar: load
    with st.sidebar:
        st.header("Load Database")
        choice = st.radio("Select source", ["Default path", "Upload JSON", "Enter path"], index=0)
        data = None
        used_path = None

        if choice == "Default path":
            used_path = DEFAULT_JSON_PATH
            if os.path.exists(used_path):
                data = load_json(used_path)
            else:
                st.warning(f"Default JSON not found: {used_path}")
        elif choice == "Upload JSON":
            up = st.file_uploader("Upload data.json", type=["json"])
            if up is not None:
                data = load_json_bytes(up.getvalue())
                used_path = up.name or "uploaded.json"
        else:
            path = st.text_input("Path to JSON file", value=DEFAULT_JSON_PATH)
            if path:
                try:
                    data = load_json(path)
                    used_path = path
                except Exception as e:
                    st.error(f"Failed to load: {e}")

    if data is None:
        st.stop()

    phases, chapters = infer_filters(data)

    st.sidebar.divider()
    st.sidebar.header("Filters")
    q = st.sidebar.text_input("Search (full-text; includes subtask tools)", "")
    phase_sel = st.sidebar.multiselect("Phase", options=phases, default=[])
    chap_sel = st.sidebar.multiselect("SRP Chapter", options=chapters, default=[])

    # Graph controls
    st.sidebar.divider()
    st.sidebar.header("Graph")
    use_agraph = st.sidebar.toggle("Use interactive engine (agraph)", value=HAS_AGRAPH and True)
    hierarchical = st.sidebar.toggle("Use hierarchical layout (recommended)", value=True)
    physics = st.sidebar.toggle("Enable physics", value=False)

    st.sidebar.divider()
    st.sidebar.header("Export")
    filtered_tasks = filter_tasks(data, q, phase_sel, chap_sel)
    export_payload = {"meta": data.get("meta", {}), "tasks": filtered_tasks}
    st.sidebar.download_button("Download filtered JSON", data=json.dumps(export_payload, indent=2),
                               file_name="nrc_licensing_filtered.json", mime="application/json")

    # ---------- Selection state & graph event gating ----------
    if "selected_id" not in st.session_state:
        st.session_state["selected_id"] = str(filtered_tasks[0]["id"]) if filtered_tasks else None
    if "_sync_from_graph" not in st.session_state:
        st.session_state["_sync_from_graph"] = False
    if "last_graph_click_id" not in st.session_state:
        st.session_state["last_graph_click_id"] = None
    if "ignore_graph_clicks" not in st.session_state:
        st.session_state["ignore_graph_clicks"] = False
    # reset the graph click memory when graph controls change
    graph_sig = f"{int(use_agraph)}-{int(hierarchical)}-{int(physics)}"
    if st.session_state.get("_graph_sig") != graph_sig:
        st.session_state["_graph_sig"] = graph_sig
        st.session_state["last_graph_click_id"] = None
        st.session_state["ignore_graph_clicks"] = False

    # Tabs
    tab_graph, tab_data = st.tabs(["📈 Graph & Details", "📊 Data Browser"])

    # ---- Graph & Details tab ----
    with tab_graph:
        col_graph, col_details = st.columns([7,5])

        with col_graph:
            st.subheader("Process Graph")

            # Render graph and capture click
            clicked = render_graph_interactive(
                filtered_tasks,
                use_agraph=use_agraph,
                hierarchical=hierarchical,
                physics=physics
            )

            # Treat graph clicks as edge-triggered events only
            new_click = clicked and (clicked != st.session_state.get("last_graph_click_id"))

            if st.session_state.get("ignore_graph_clicks", False):
                # While the user is driving via dropdown, ignore repeats
                if new_click:
                    # user clicked a *different* node → hand control back to graph
                    st.session_state["ignore_graph_clicks"] = False
                    st.session_state["last_graph_click_id"] = clicked
                    if clicked != st.session_state.get("selected_id"):
                        st.session_state["selected_id"] = clicked
                        st.session_state["_sync_from_graph"] = True
            else:
                # Graph is the driver only when a *new* click occurs
                if new_click:
                    st.session_state["last_graph_click_id"] = clicked
                    if clicked != st.session_state.get("selected_id"):
                        st.session_state["selected_id"] = clicked
                        st.session_state["_sync_from_graph"] = True

            st.caption(
                f"Layout: {'Hierarchical' if hierarchical else 'Free'} • "
                f"Physics: {'On' if physics else 'Off'} • "
                f"Engine: {'agraph' if (use_agraph and HAS_AGRAPH) else 'pyvis'}"
            )
            if not (use_agraph and HAS_AGRAPH):
                st.info("Tip: Install `streamlit-agraph` for click-to-select.")

        with col_details:
            st.subheader("Task Details")
            options = [f"{t.get('id')} — {t.get('name')}" for t in filtered_tasks]
            if options:
                id_to_idx = {str(t.get("id")): i for i, t in enumerate(filtered_tasks)}
                cur_id = st.session_state.get("selected_id")
                cur_idx = id_to_idx.get(str(cur_id), 0)
                current_option_label = options[cur_idx]

                # One-shot sync from graph → selectbox
                if st.session_state.get("_sync_from_graph"):
                    st.session_state["task_select"] = current_option_label
                    st.session_state["_sync_from_graph"] = False
                elif "task_select" not in st.session_state:
                    st.session_state["task_select"] = current_option_label

                # When dropdown changes, let dropdown drive and ignore stale graph selection
                def _on_select_change():
                    lbl = st.session_state["task_select"]
                    sid = lbl.split(" — ", 1)[0].strip()
                    st.session_state["selected_id"] = sid
                    st.session_state["ignore_graph_clicks"] = True  # prevent immediate reversion

                sel_label = st.selectbox(
                    "Select a task",
                    options=options,
                    key="task_select",
                    on_change=_on_select_change
                )

                # Show details based on the single source of truth
                selected_id = st.session_state.get("selected_id")
                task = next((t for t in filtered_tasks if str(t.get("id")) == str(selected_id)), None)
                if task:
                    details_block(task)
            else:
                st.write("No tasks match the current filters/search.")

        st.divider()
        st.caption(f"Source: {used_path if used_path else '(uploaded)'} • "
                   f"{data.get('meta',{}).get('name','')} v{data.get('meta',{}).get('version','')} • "
                   f"Generated: {data.get('meta',{}).get('generated','')}")

    # ---- Data Browser tab ----
    with tab_data:
        st.subheader("Excel-like Database Browser & Filters")
        if not HAS_PANDAS:
            st.warning("Install pandas to enable the Data Browser: `pip install pandas`")
        else:
            df_tasks, df_reqs, df_tools, df_subtasks = build_dataframes(data)

            with st.expander("Filters", expanded=True):
                f_phase = st.multiselect("Phase", options=sorted([p for p in df_tasks["phase"].dropna().unique()]),
                                         default=[])
                req_types_avail = sorted([x for x in df_reqs["req_type"].dropna().unique()]) if not df_reqs.empty else []
                f_req_type = st.multiselect("Requirement Type", options=req_types_avail, default=[])
                f_req_cite = st.text_input("Requirement Cite contains", "")
                tool_types_avail = sorted([x for x in df_tools["tool_type"].dropna().unique()]) if not df_tools.empty else []
                f_tool_type = st.multiselect("Tool Type", options=tool_types_avail, default=[])
                f_tool_name = st.text_input("Tool Name contains", "")
                f_task_text = st.text_input("Task name/description contains", "")

            def apply_task_filters(df: pd.DataFrame) -> pd.DataFrame:
                out = df.copy()
                if f_phase:
                    out = out[out["phase"].isin(f_phase)]
                if f_task_text:
                    patt = re.escape(f_task_text)
                    mask = out["name"].str.contains(patt, case=False, na=False) | out["description"].str.contains(patt, case=False, na=False)
                    out = out[mask]
                if f_req_type:
                    patt = "|".join([re.escape(x) for x in f_req_type])
                    out = out[out["requirements_types"].str.contains(patt, case=False, na=False)]
                if f_tool_type:
                    patt = "|".join([re.escape(x) for x in f_tool_type])
                    out = out[out["tool_types"].str.contains(patt, case=False, na=False)]
                if f_tool_name:
                    out = out[out["tool_names"].str.contains(re.escape(f_tool_name), case=False, na=False)]
                return out

            def apply_req_filters(df: pd.DataFrame) -> pd.DataFrame:
                out = df.copy()
                if f_phase:
                    out = out[out["phase"].isin(f_phase)]
                if f_req_type:
                    out = out[out["req_type"].isin(f_req_type)]
                if f_req_cite:
                    out = out[out["req_cite"].str.contains(re.escape(f_req_cite), case=False, na=False)]
                return out

            def apply_tool_filters(df: pd.DataFrame) -> pd.DataFrame:
                out = df.copy()
                if f_phase:
                    out = out[out["phase"].isin(f_phase)]
                if f_tool_type:
                    out = out[out["tool_type"].isin(f_tool_type)]
                if f_tool_name:
                    out = out[out["tool_name"].str.contains(re.escape(f_tool_name), case=False, na=False)]
                return out

            subtabs = st.tabs(["Tasks (Aggregated)", "Requirements (Exploded)", "Tools (Exploded)", "Subtasks (Exploded)"])

            with subtabs[0]:
                tdf = apply_task_filters(df_tasks)
                st.dataframe(tdf, use_container_width=True, height=500)
                df_download_button(tdf, "Download tasks (CSV)", "tasks_aggregated.csv")

            with subtabs[1]:
                rdf = apply_req_filters(df_reqs)
                st.dataframe(rdf, use_container_width=True, height=500)
                df_download_button(rdf, "Download requirements (CSV)", "requirements_exploded.csv")

            with subtabs[2]:
                xdf = apply_tool_filters(df_tools)
                st.dataframe(xdf, use_container_width=True, height=500)
                df_download_button(xdf, "Download tools (CSV)", "tools_exploded.csv")

            with subtabs[3]:
                sdf = df_subtasks.copy()
                if f_phase:
                    sdf = sdf[sdf["phase"].isin(f_phase)]
                st.dataframe(sdf, use_container_width=True, height=500)
                df_download_button(sdf, "Download subtasks (CSV)", "subtasks_exploded.csv")


if __name__ == "__main__":
    main()
