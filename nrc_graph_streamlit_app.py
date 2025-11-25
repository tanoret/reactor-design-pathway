# NRC Licensing Process — Streamlit Graph Viewer (optimized)
# ---------------------------------------------------------
# Improvements:
#   - Cached search index and DataFrames
#   - Sidebar toggles for interactive graph / physics
#   - Neighborhood (N-hop) graph subset with max-node cap
#   - Lighter PyVis defaults (physics off, config panel off)
#
# Usage:
#   pip install -r requirements_streamlit.txt
#   # or:
#   pip install streamlit streamlit-agraph pyvis networkx pandas
#   streamlit run nrc_graph_streamlit_app.py

import json
import os
import re
from collections import deque
from typing import Dict, Any, List, Tuple, Optional

import streamlit as st

# Optional libs for graph rendering (PyVis fallback)
try:
    import networkx as nx  # noqa: F401
    from pyvis.network import Network  # noqa: F401
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

# --------- CACHING: search index ---------
@st.cache_data(show_spinner=False)
def build_search_index(tasks: List[Dict[str, Any]]) -> Dict[str, str]:
    """One-time build of lowercase text blobs for fast filtering."""
    return {str(t.get("id", "")): text_blob(t) for t in tasks}

def filter_tasks(
    data: Dict[str, Any],
    query: str,
    phases_sel: List[str],
    chapters_sel: List[str],
    index: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    tasks = data.get("tasks", [])
    q = (query or "").strip().lower()
    res = []
    for t in tasks:
        if phases_sel and t.get("phase") not in phases_sel:
            continue
        if chapters_sel and (t.get("srpChapter") or "") not in chapters_sel:
            continue
        if q:
            blob = (index or {}).get(str(t.get("id")))
            if blob is None:
                blob = text_blob(t)  # fallback if index missing
            if q not in blob:
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
# Graph helpers: subset
# -------------------------
def subset_tasks_by_hops(tasks: List[Dict[str, Any]], center_id: Optional[str], hops: int) -> List[Dict[str, Any]]:
    """Return tasks within N hops (both directions) from center_id, preserving original order."""
    if not center_id or hops <= 0:
        return tasks

    by_id = {str(t.get("id")): t for t in tasks}
    if center_id not in by_id:
        return tasks

    # Build adjacency (parents via deps; children via reverse deps)
    parents = {tid: set(map(str, (by_id[tid].get("deps") or []))) for tid in by_id}
    children = {tid: set() for tid in by_id}
    for t in tasks:
        tid = str(t.get("id"))
        for d in (t.get("deps") or []):
            d = str(d)
            if d in children:
                children[d].add(tid)

    seen = {center_id}
    dq = deque([(center_id, 0)])
    while dq:
        nid, dist = dq.popleft()
        if dist == hops:
            continue
        for nbr in parents.get(nid, set()) | children.get(nid, set()):
            if nbr in by_id and nbr not in seen:
                seen.add(nbr)
                dq.append((nbr, dist + 1))

    return [t for t in tasks if str(t.get("id")) in seen]


# -------------------------
# Graph rendering (PyVis)
# -------------------------
def build_graph_html(tasks: List[Dict[str, Any]], physics: bool = False) -> str:
    if not HAS_GRAPH:
        return "<div style='padding:16px;font-family:system-ui'>Install dependencies to render graph: <code>pip install networkx pyvis</code></div>"
    import networkx as nx
    from pyvis.network import Network

    G = nx.DiGraph()
    for t in tasks:
        nid = str(t.get("id"))
        label = f"{t.get('id','')}  {t.get('name','')}".strip()
        title = f"<b>{t.get('id','')}</b> — {t.get('name','')}"
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
        G.add_node(nid, label=label, title=title_html, color=color_for_phase(phase), shape="box")

    for t in tasks:
        for d in t.get("deps") or []:
            if str(d) in G.nodes and str(t.get("id")) in G.nodes:
                G.add_edge(str(d), str(t.get("id")))

    net = Network(height="700px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222")
    net.set_options(f"""
    {{
      "layout": {{"improvedLayout": true}},
      "physics": {{
        "enabled": {str(physics).lower()},
        "solver": "barnesHut",
        "stabilization": {{"enabled": true, "fit": true}},
        "barnesHut": {{"gravitationalConstant": -30000, "centralGravity": 0.2, "springLength": 160, "springConstant": 0.05, "damping": 0.8}}
      }},
      "interaction": {{"hover": true, "tooltipDelay": 100, "navigationButtons": true, "keyboard": {{"enabled": true}}}},
      "edges": {{"arrows": {{"to": {{"enabled": true}}}}}},
      "configure": {{"enabled": false}}
    }}
    """)
    net.from_nx(G)
    return net.generate_html()


# -------------------------
# Graph rendering (Agraph) with click feedback
# -------------------------
def render_graph_interactive(
    tasks: List[Dict[str, Any]],
    physics: bool = False,
    use_agraph: bool = True
) -> Optional[str]:
    """
    If streamlit-agraph is available, render an interactive graph and return the clicked node id.
    Otherwise, fall back to PyVis HTML (no click feedback) and return None.
    """
    if not tasks:
        return None

    if HAS_AGRAPH and use_agraph:
        nodes: List[Node] = []
        edges: List[Edge] = []
        for t in tasks:
            nid = str(t.get("id"))
            label = f"{t.get('id','')} — {t.get('name','')}"
            phase = t.get("phase") or ""
            desc = t.get("description") or ""
            nodes.append(
                Node(
                    id=nid,
                    label=label,
                    title=f"<b>{nid}</b><br/>{desc}",
                    size=18,
                    shape="box",
                    color=color_for_phase(phase),
                )
            )
        for t in tasks:
            for d in t.get("deps") or []:
                edges.append(Edge(source=str(d), target=str(t.get("id"))))

        config = Config(
            width=900, height=700, directed=True, physics=physics,
            hierarchical=False,  # Set True if you want layered layout
            nodeHighlightBehavior=True
        )
        ret = agraph(nodes=nodes, edges=edges, config=config)
        if isinstance(ret, dict):
            # 'selected_node', 'selected_nodes', 'clicked_node'
            cand = ret.get("selected_node") or ret.get("clicked_node")
            if not cand:
                nodes_list = ret.get("selected_nodes") or []
                cand = nodes_list[0] if nodes_list else None
            return str(cand) if cand else None
        if isinstance(ret, (list, tuple)) and ret:
            return str(ret[0])
        if isinstance(ret, str) and ret:
            return ret
        return None

    # PyVis fallback (no click feedback)
    html = build_graph_html(tasks, physics=physics)
    st.components.v1.html(html, height=740, scrolling=True)
    st.info("Tip: For click-to-select, install `streamlit-agraph` (pip) to enable interactive linking.")
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

    st.download_button(
        "Download this task as JSON",
        data=json.dumps(task, indent=2),
        file_name=f"task_{task.get('id','')}.json",
        mime="application/json"
    )


# -------------------------
# Data Browser helpers
# -------------------------
@st.cache_data(show_spinner=False)
def build_dataframes(data: Dict[str, Any]):
    """
    Build aggregated + exploded DataFrames:
      - df_tasks: one row per task (aggregated fields)
      - df_reqs: one row per requirement
      - df_tools: one row per (subtask tool)
      - df_subtasks: one row per subtask
    """
    if not HAS_PANDAS:
        return None, None, None, None

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

        # Exploded requirements
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
        # Exploded subtasks & tools
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

    df_tasks = pd.DataFrame(rows_tasks)
    df_reqs = pd.DataFrame(rows_reqs)
    df_tools = pd.DataFrame(rows_tools)
    df_subtasks = pd.DataFrame(rows_subtasks)
    return df_tasks, df_reqs, df_tools, df_subtasks

def df_download_button(df, label: str, filename: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv, file_name=filename, mime="text/csv")


# -------------------------
# App
# -------------------------
def main():
    st.title("NRC Licensing Process — JSON Graph Viewer")
    st.caption("Loads a JSON database and renders a linked task graph with details and filters. Now with click-to-select (agraph) and a Data Browser tab.")

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

    # Build once and reuse
    search_index = build_search_index(data.get("tasks", []))
    phases, chapters = infer_filters(data)

    st.sidebar.divider()
    st.sidebar.header("Filters")
    q = st.sidebar.text_input("Search (full-text; includes subtask tools)", "")
    phase_sel = st.sidebar.multiselect("Phase", options=phases, default=[])
    chap_sel = st.sidebar.multiselect("SRP Chapter", options=chapters, default=[])

    # Performance controls
    st.sidebar.divider()
    st.sidebar.header("Performance")
    use_interactive_default = True and HAS_AGRAPH
    use_interactive = st.sidebar.toggle("Use interactive graph (agraph if available)", value=use_interactive_default)
    physics_on = st.sidebar.toggle("Enable graph physics", value=False)
    hops = st.sidebar.slider("Neighborhood hops around selection", 0, 3, 1)
    max_nodes = st.sidebar.slider("Max nodes to render", 50, 2000, 400, 50)

    st.sidebar.divider()
    st.sidebar.header("Export")
    filtered_tasks = filter_tasks(data, q, phase_sel, chap_sel, index=search_index)
    export_payload = {"meta": data.get("meta", {}), "tasks": filtered_tasks}
    st.sidebar.download_button("Download filtered JSON", data=json.dumps(export_payload, indent=2),
                               file_name="nrc_licensing_filtered.json", mime="application/json")

    # Maintain current selection in session state
    if "selected_id" not in st.session_state:
        st.session_state["selected_id"] = str(filtered_tasks[0]["id"]) if filtered_tasks else None

    # Tabs
    tab_graph, tab_data = st.tabs(["📈 Graph & Details", "📊 Data Browser"])

    # ---- Graph & Details tab ----
    with tab_graph:
        col_graph, col_details = st.columns([7,5])
        with col_graph:
            st.subheader("Process Graph")

            # Reduce to neighborhood & cap count
            graph_tasks = filtered_tasks
            if st.session_state.get("selected_id") and hops > 0:
                graph_tasks = subset_tasks_by_hops(filtered_tasks, st.session_state["selected_id"], hops)
            if len(graph_tasks) > max_nodes:
                graph_tasks = graph_tasks[:max_nodes]

            # Use interactive agraph if possible, else PyVis fallback
            selected_from_graph = render_graph_interactive(
                graph_tasks, physics=physics_on, use_agraph=use_interactive
            )
            if selected_from_graph:
                st.session_state["selected_id"] = selected_from_graph

            st.caption(f"Rendering {len(graph_tasks)} nodes "
                       f"(hops={hops}, physics={'on' if physics_on else 'off'}, "
                       f"interactive={'on' if use_interactive else 'off'}).")
            st.markdown(":memo: **Tip:** Click a node (agraph) to load its details. Turn physics on only when needed for layout.")

        with col_details:
            st.subheader("Task Details")
            options = [f"{t.get('id')} — {t.get('name')}" for t in filtered_tasks]
            if options:
                # Compute index from session_state selection (if still present after filtering)
                sel_id = st.session_state.get("selected_id")
                id_to_idx = {str(t.get("id")): i for i, t in enumerate(filtered_tasks)}
                pre_idx = id_to_idx.get(str(sel_id), 0)
                sel_label = st.selectbox("Select a task", options=options, index=pre_idx, key="task_select")
                selected_id = sel_label.split(" — ", 1)[0].strip()
                st.session_state["selected_id"] = selected_id  # keep in sync with dropdown

                # Show details
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
                # Common
                f_phase = st.multiselect("Phase", options=sorted([p for p in df_tasks["phase"].dropna().unique()]),
                                         default=[])
                # Requirements filters
                req_types_avail = sorted([x for x in df_reqs["req_type"].dropna().unique()]) if not df_reqs.empty else []
                f_req_type = st.multiselect("Requirement Type", options=req_types_avail, default=[])
                f_req_cite = st.text_input("Requirement Cite contains", "")
                # Tool filters
                tool_types_avail = sorted([x for x in df_tools["tool_type"].dropna().unique()]) if not df_tools.empty else []
                f_tool_type = st.multiselect("Tool Type", options=tool_types_avail, default=[])
                f_tool_name = st.text_input("Tool Name contains", "")
                # Task text search
                f_task_text = st.text_input("Task name/description contains", "")

            # Filter functions
            def apply_task_filters(df):
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

            def apply_req_filters(df):
                out = df.copy()
                if f_phase:
                    out = out[out["phase"].isin(f_phase)]
                if f_req_type:
                    out = out[out["req_type"].isin(f_req_type)]
                if f_req_cite:
                    out = out[out["req_cite"].str.contains(re.escape(f_req_cite), case=False, na=False)]
                return out

            def apply_tool_filters(df):
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
