# NRC Licensing Process — Streamlit Graph Viewer

Interactive viewer for the NRC licensing JSON database: filter, explore a dependency graph, click a node to open details, and browse a full Excel-like dataset (requirements, tools, subtasks).

## Features
- **Graph:** Dependency graph with hover tooltips and (if `streamlit-agraph` is installed) click-to-select.
- **Details:** Full task details, subtasks, tools, requirements, deliverables, acceptance criteria.
- **Data Browser:** Excel-like tables for tasks, requirements, tools, and subtasks, with rich filters and CSV export.
- **Offline PyVis:** If using PyVis, configure it to inline JS/CSS (no CDN hang).

## Local run
```bash
pip install -r requirements.txt
streamlit run nrc_graph_streamlit_app.py
