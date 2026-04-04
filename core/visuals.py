import json
import os
import re
import numpy as np
import pandas as pd
import plotly.colors
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import py3Dmol
except Exception:
    py3Dmol = None


def _apply_transparent_background(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _plot_sequence_colormap(df, title: str, value_col: str, symmetric: bool, legend_title: str, colors):
    values = df[value_col].to_numpy(dtype=float)
    residues = df["residue"].astype(str).tolist()
    positions = df["position"].to_numpy(dtype=int)
    length = len(residues)

    if length == 0:
        st.info("No residues available for visualization.")
        return

    colorscale = [[0.0, colors[0]], [0.5, colors[1]], [1.0, colors[2]]]

    if symmetric:
        max_abs = np.max(np.abs(values)) if values.size else 1.0
        z = np.zeros_like(values) if max_abs == 0 else (values / max_abs)
        z = z.reshape(1, -1)
        zmin, zmax = -1.0, 1.0
        tickvals = [-1, 0, 1]
        ticktext = ["neg", "0", "pos"]
    else:
        vmax = np.max(values) if values.size else 1.0
        z = np.zeros_like(values) if vmax <= 0 else np.clip(values / vmax, 0.0, 1.0)
        z = z.reshape(1, -1)
        zmin, zmax = 0.0, 1.0
        tickvals = [0, 0.5, 1]
        ticktext = ["low", "med", "high"]

    bg_color = "rgba(0,0,0,0)"

    # Fixed-size cells make letters and indices easy to read.
    cell_px = 13
    margin = dict(l=0, r=0, t=30, b=130)
    inner_w = length * cell_px
    inner_h = cell_px
    fig_w = inner_w + margin["l"] + margin["r"]
    fig_h = inner_h + margin["t"] + margin["b"]

    x_idx = list(range(length))

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_idx,
            y=[0],
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale,
            xgap=1,
            ygap=1,
            showscale=False,
            customdata=np.array([[str(p) for p in positions]]),
            hovertemplate=(
                "<b>Position:</b> %{customdata}<br>"
                "<b>Residue:</b> %{text}<br>"
                f"<b>{value_col.capitalize()}:</b> %{{z:.4f}}<extra></extra>"
            ),
            text=np.array([residues]),
        )
    )

    annotations = []
    for i in x_idx:
        annotations.append(
            dict(
                x=i,
                y=0.10,
                text=residues[i],
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                font=dict(size=11),
            )
        )
        annotations.append(
            dict(
                xref="x",
                yref="paper",
                x=i,
                y=-0.55,
                text=str(positions[i]),
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                font=dict(size=6.5),
                textangle=-90,
            )
        )

    fig.update_layout(
        title=title,
        width=fig_w,
        height=fig_h,
        margin=margin,
        annotations=annotations,
    )
    _apply_transparent_background(fig)

    fig.update_xaxes(
        range=[-0.5, length - 0.5],
        showgrid=False,
        showticklabels=False,
        ticks="",
        ticklen=0,
        zeroline=False,
        fixedrange=True,
    )
    fig.update_yaxes(
        range=[-0.5, 0.5],
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        fixedrange=True,
        scaleanchor="x",
        scaleratio=1,
    )

    # Render inside a scrollable frame so long sequences stay readable.
    html = fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": False})
    
    # Wrap with custom CSS/JS to hide scrollbars but enable wheel scrolling
    wrapper_html = f"""
    <div id="plot-shell" style="overflow: hidden; width: 100%;">
        <div id="plot-container" style="overflow-x: auto; padding: 4px 0;
                                        scrollbar-width: none; -ms-overflow-style: none;
                                        cursor: grab; user-select: none;">
            {html}
        </div>
    </div>
    <script>
        const plotShell = document.getElementById('plot-shell');
        const plotContainer = document.getElementById('plot-container');
        let isDragging = false;
        let dragStartX = 0;
        let dragStartScrollLeft = 0;

        plotContainer.addEventListener('wheel', function(e) {{
            if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {{
                e.preventDefault();
                this.scrollLeft += e.deltaY;
            }}
        }}, {{ passive: false }});

        plotContainer.style.scrollBehavior = 'smooth';

        plotContainer.addEventListener('mousedown', function(e) {{
            isDragging = true;
            dragStartX = e.clientX;
            dragStartScrollLeft = plotContainer.scrollLeft;
            plotContainer.style.cursor = 'grabbing';
            plotContainer.style.scrollBehavior = 'auto';
            e.preventDefault();
        }});

        window.addEventListener('mousemove', function(e) {{
            if (!isDragging) return;
            const dx = e.clientX - dragStartX;
            plotContainer.scrollLeft = dragStartScrollLeft - dx;
        }});

        window.addEventListener('mouseup', function() {{
            if (!isDragging) return;
            isDragging = false;
            plotContainer.style.cursor = 'grab';
            plotContainer.style.scrollBehavior = 'smooth';
        }});

        plotContainer.addEventListener('mouseleave', function() {{
            if (!isDragging) return;
            isDragging = false;
            plotContainer.style.cursor = 'grab';
            plotContainer.style.scrollBehavior = 'smooth';
        }});

        function getPlotNode() {{
            return plotContainer.querySelector('.js-plotly-plot');
        }}

        function toggleDoubleClickZoom() {{
            const plotNode = getPlotNode();
            if (!plotNode) return;

            const isZoomed = plotNode.getAttribute('data-zoomed') === '1';
            if (isZoomed) {{
                plotNode.style.transform = 'scale(1)';
                plotNode.style.transformOrigin = 'top left';
                plotNode.setAttribute('data-zoomed', '0');
            }} else {{
                plotNode.style.transform = 'scale(1.5)';
                plotNode.style.transformOrigin = 'top left';
                plotNode.setAttribute('data-zoomed', '1');
            }}
        }}

        setTimeout(function() {{
            const plotNode = getPlotNode();
            if (plotNode) {{
                plotNode.addEventListener('dblclick', function(e) {{
                    e.preventDefault();
                    toggleDoubleClickZoom();
                }});
            }}
        }}, 120);
    </script>
    <style>
        #plot-container::-webkit-scrollbar {{ display: none; }}
    </style>
    """
    
    # Build fixed colorbar HTML
    colorbar_html = f"""
    <div style="position: fixed; bottom: 2px; right: 16px; z-index: 9999;
                 background-color: {bg_color}; padding: 8px;
                 border: 1px solid rgba(128,128,128,0.3); border-radius: 4px;
                 color: inherit;">
        <div style="font-size: 11px; font-weight: 400;
                    margin-bottom: 4px; text-align: center;">
            {legend_title}
        </div>
        <svg width="220" height="16">
            <defs>
                <linearGradient id="cbar" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="{colors[0]}" />
                    <stop offset="50%" stop-color="{colors[1]}" />
                    <stop offset="100%" stop-color="{colors[2]}" />
                </linearGradient>
            </defs>
            <rect x="0" y="0" width="220" height="16" fill="url(#cbar)" stroke="currentColor" stroke-width="0.5"/>
        </svg>
        <div style="display: flex; justify-content: space-between; font-size: 9px;
                    margin-top: 2px;">
            <span>{ticktext[0]}</span>
            <span>{ticktext[1]}</span>
            <span>{ticktext[2]}</span>
        </div>
    </div>
    """
    
    wrapper_final = wrapper_html + colorbar_html
    st.components.v1.html(wrapper_final, height=fig_h + 20, scrolling=False)


def plot_importance(df, title: str):
    theme_type = str(getattr(getattr(st.context, "theme", None), "type", "light")).lower()
    is_dark = theme_type == "dark"
    
    # Theme-aware colorscale for better contrast: red -> neutral -> green
    if is_dark:
        # Dark theme: use lighter neutral gray for better visibility
        colors = ["#ff4444", "#0e1117", "#44dd44"]
    else:
        # Light theme: use white or light gray for neutral
        colors = ["#cc0000", "#f5f5f5", "#00aa00"]
    
    _plot_sequence_colormap(
        df=df,
        title=title,
        value_col="score",
        symmetric=True,
        legend_title="Contribution to Prediction",
        colors=colors,
    )

def plot_attention(df, title: str):
    theme_type = str(getattr(getattr(st.context, "theme", None), "type", "light")).lower()
    is_dark = theme_type == "dark"
    
    # Theme-aware colorscale for better contrast: start from visible color, go to contrasting
    if is_dark:
        # Dark theme: start from light color, go to dark saturated color
        colors = ["#0e1117", "#2C224A", "#9058d4"]
    else:
        # Light theme: start from light color, go to dark saturated color
        colors = ["#f0e6ff", "#dfd0f8", "#9058d4"]
    
    _plot_sequence_colormap(
        df=df,
        title=title,
        value_col="attention",
        symmetric=False,
        legend_title="Attention Weight",
        colors=colors,
    )


def plot_top_attributes(top_attrs: pd.DataFrame, title: str = "Top 10 attributes (5 most positive, 5 most negative)"):
    if top_attrs.empty:
        st.info("No residue attributes available.")
        return

    st.markdown(f"**{title}**")
    top_attrs_display = top_attrs[["position", "residue", "score", "contribution"]].copy()
    top_attrs_display["raw_score"] = top_attrs_display["score"]
    max_abs_score = float(top_attrs_display["score"].abs().max()) if not top_attrs_display.empty else 0.0
    top_attrs_display["display_score"] = (
        top_attrs_display["score"] / max_abs_score if max_abs_score > 0 else 0.0
    )
    top_attrs_display["label"] = top_attrs_display.apply(
        lambda r: f"{r['residue']}:{int(r['position'])}", axis=1
    )
    top_attrs_display = top_attrs_display.sort_values("raw_score", ascending=True)

    base_colors = {
        "Positive": "#22c55e",
        "Negative": "#ef4444",
        "Neutral": "#94a3b8",
    }

    fig_top_attrs = px.bar(
        top_attrs_display,
        x="display_score",
        y="label",
        orientation="h",
        color="contribution",
        color_discrete_map=base_colors,
        hover_data={
            "position": True,
            "residue": True,
            "raw_score": ":.4f",
            "display_score": ":.3f",
            "contribution": True,
            "label": False,
        },
    )

    translucent_fill = {
        "Positive": "rgba(34, 197, 94, 0.22)",
        "Negative": "rgba(239, 68, 68, 0.22)",
        "Neutral": "rgba(148, 163, 184, 0.18)",
    }
    for contrib, outline in base_colors.items():
        fig_top_attrs.update_traces(
            selector={"name": contrib},
            marker=dict(
                color=translucent_fill[contrib],
                line=dict(color=outline, width=1.6),
                cornerradius=2,
            ),
            opacity=0.95,
        )

    fig_top_attrs.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Relative IG Score",
        yaxis_title="Residue@Position",
        legend_title_text="",
        bargap=0.35,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
    )
    fig_top_attrs.update_xaxes(
        gridcolor="rgba(148, 163, 184, 0.22)",
        zerolinecolor="rgba(148, 163, 184, 0.35)",
        linecolor="rgba(148, 163, 184, 0.25)",
        tickcolor="rgba(148, 163, 184, 0.25)",
    )
    fig_top_attrs.update_yaxes(
        gridcolor="rgba(148, 163, 184, 0.08)",
        linecolor="rgba(148, 163, 184, 0.25)",
    )
    fig_top_attrs.add_vline(x=0, line_width=1, line_dash="dash", line_color="rgba(148, 163, 184, 0.7)")
    _apply_transparent_background(fig_top_attrs)
    st.plotly_chart(fig_top_attrs, width="stretch")


def plot_residue_boxplot(df, value_col: str, title: str, y_title: str, key: str = None):
    labels = [f"{int(pos)}:{res}" for pos, res in zip(df["position"], df["residue"])]
    values = df[value_col].astype(float).to_numpy()

    has_negative = np.min(values) < 0 if values.size else False
    if has_negative:
        max_abs = max(np.max(np.abs(values)), 1e-6)
        normalized_values = values / max_abs
        marker = dict(
            color=normalized_values,
            cmin=-1,
            cmax=1,
        )
    else:
        vmax = max(np.max(values), 1e-6) if values.size else 1.0
        normalized_values = values / vmax
        marker = dict(
            color=normalized_values,
            cmin=0,
            cmax=1,
        )

    fig = go.Figure(
        data=go.Bar(
            x=labels,
            y=normalized_values,
            marker=marker,
            hovertemplate=(
                "<b>Residue:</b> %{x}<br>"
                f"<b>{y_title}:</b> %{{y:.4f}}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Residue (position:AA)",
        yaxis_title=f"{y_title}",
        width=1200,
        height=460,
        margin=dict(l=40, r=20, t=50, b=140),
    )
    _apply_transparent_background(fig)
    fig.update_xaxes(tickangle=-70)
    st.plotly_chart(fig, width="stretch", key=key)


def _apply_residue_importance_to_pdb(pdb_text: str, residue_importance=None):
    if residue_importance is None:
        return pdb_text

    if isinstance(residue_importance, pd.DataFrame):
        if "position" not in residue_importance:
            return pdb_text
        value_column = "normalized_score" if "normalized_score" in residue_importance.columns else "score"
        score_map = {int(row["position"]): float(row[value_column]) for _, row in residue_importance.iterrows()}
    else:
        score_map = {int(position): float(score) for position, score in residue_importance}

    lines = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 66:
            try:
                residue_number = int(line[22:26])
            except ValueError:
                lines.append(line)
                continue
            score = score_map.get(residue_number)
            if score is None:
                lines.append(line)
                continue
            score = max(-1.0, min(1.0, float(score)))
            line = f"{line[:60]}{score:6.2f}{line[66:]}"
        lines.append(line)
    return "\n".join(lines)


def show_structure_viewer(pdb_path, residue_importance=None, style_mode: str = "cartoon"):
    if py3Dmol is None:
        st.info("Install py3Dmol to view structures inline. The PDB can still be downloaded.")
        return

    theme_type = str(getattr(getattr(st.context, "theme", None), "type", "light")).lower()
    is_dark = theme_type == "dark"
    viewer_bg = "rgba(0,0,0,0)"
    legend_bg = "rgba(17, 24, 39, 0.88)" if is_dark else "rgba(255, 255, 255, 0.94)"
    legend_text = "#e5e7eb" if is_dark else "#111827"
    tooltip_bg = "#111827" if is_dark else "#111827"
    imp_low = "#ff4444" if is_dark else "#cc0000"
    imp_mid = "#f5f5f5"
    imp_high = "#44dd44" if is_dark else "#00aa00"

    pdb_text = pdb_path.read_text() if hasattr(pdb_path, "read_text") else str(pdb_path)
    pdb_text = _apply_residue_importance_to_pdb(pdb_text, residue_importance)

    style_key = str(style_mode).lower()
    style_name_map = {
        "sticks": "stick",
        "cartoon": "cartoon",
        "line": "line",
        "sphere": "sphere",
    }
    style_name = style_name_map.get(style_key, "stick")
    style_spec = {style_name: {}}

    view = py3Dmol.view(width=900, height=700)
    view.addModel(pdb_text, "pdb")
    view.setStyle({}, style_spec)
    view.zoomTo()

    html = view._make_html()
    viewer_match = re.search(r"viewer_\d+", html)
    if viewer_match is not None:
        viewer_var = viewer_match.group(0)
        hover_js = f"""
<script>
(function() {{
  const viewerKey = "{viewer_var}";
  function installHover(attempt) {{
    const viewer = window[viewerKey];
    if (!viewer) {{
      if (attempt < 60) setTimeout(function() {{ installHover(attempt + 1); }}, 100);
      return;
    }}
        viewer.setBackgroundColor("{viewer_bg}", 0.0);

        function hexToRgb(hex) {{
            const h = hex.replace('#', '');
            return {{
                r: parseInt(h.substring(0, 2), 16),
                g: parseInt(h.substring(2, 4), 16),
                b: parseInt(h.substring(4, 6), 16)
            }};
        }}

        function rgbToInt(rgb) {{
            return (rgb.r << 16) + (rgb.g << 8) + rgb.b;
        }}

        function lerpColor(c1, c2, t) {{
            return {{
                r: Math.round(c1.r + (c2.r - c1.r) * t),
                g: Math.round(c1.g + (c2.g - c1.g) * t),
                b: Math.round(c1.b + (c2.b - c1.b) * t)
            }};
        }}

        const neg = hexToRgb("{imp_low}");
        const mid = hexToRgb("{imp_mid}");
        const pos = hexToRgb("{imp_high}");

        const atoms = viewer.selectedAtoms({{}}) || [];
        for (let i = 0; i < atoms.length; i++) {{
            const atom = atoms[i];
            const b = (atom.b !== undefined && atom.b !== null && !isNaN(atom.b)) ? Math.max(-1, Math.min(1, Number(atom.b))) : 0;
            const rgb = b < 0 ? lerpColor(mid, neg, Math.abs(b)) : lerpColor(mid, pos, b);
            atom.color = rgbToInt(rgb);
        }}

    viewer.setHoverable(
      {{}},
      true,
      function(atom, v) {{
        if (!atom || atom.label) return;
        const score = (atom.b !== undefined && atom.b !== null && !isNaN(atom.b)) ? Number(atom.b).toFixed(3) : "N/A";
        const residueName = atom.resn || "UNK";
        const residuePos = atom.resi !== undefined ? atom.resi : "?";
        atom.label = v.addLabel(
          residueName + " (" + residuePos + ") Importance: " + score,
          {{
            position: atom,
            showBackground: true,
                        backgroundColor: "{tooltip_bg}",
            backgroundOpacity: 0.72,
            fontColor: "white",
            fontSize: 12,
            inFront: true
          }}
        );
      }},
      function(atom, v) {{
        if (!atom || !atom.label) return;
        v.removeLabel(atom.label);
        delete atom.label;
      }}
    );
    viewer.render();
  }}
  installHover(0);
}})();
</script>
"""
        html += hover_js

        legend_html = f"""
<div style="
    position: absolute;
    top: 8px;
    right: 8px;
    width: 220px;
    padding: 6px 8px;
    border-radius: 8px;
    background: {legend_bg};
    border: 1px solid rgba(148, 163, 184, 0.35);
    color: {legend_text};
    font-size: 10px;
    line-height: 1.2;
    font-family: sans-serif;
    z-index: 20;
">
    <div style="font-weight: 600; margin-bottom: 4px;">Residue Importance</div>
    <div style="height: 8px; border-radius: 4px; background: linear-gradient(90deg, {imp_low} 0%, {imp_mid} 50%, {imp_high} 100%);"></div>
    <div style="display: flex; justify-content: space-between; margin-top: 3px;">
        <span>neg</span>
        <span>0</span>
        <span>pos</span>
    </div>
</div>
"""

        combined_html = f"""
<div style="position: relative; width: 100%; height: 500px; background: transparent;">
{html}
{legend_html}
</div>
"""

        st.components.v1.html(combined_html, height=520, scrolling=False)


def visualize_sequence_residue_embeddings(
    ids,
    residues,
    embeddings,
    max_plot_sequences=None,
    mode="pca",
    n_pcs=3,
):
    """Visualize sequence residue embeddings using PCA or raw dimensions.
    
    Args:
        ids: Sequence identifiers
        residues: Residue characters per sequence
        embeddings: Embeddings tensor (num_sequences, num_residues, embedding_dim)
        max_plot_sequences: Max sequences to plot (auto-infer if None)
        mode: 'pca' or 'raw_dims'
        n_pcs: Number of principal components for PCA mode
        mode: 'pca' or 'raw_dims'
        n_pcs: Number of principal components for PCA mode
    """
    # Theme detection for colorscales
    theme_type = str(getattr(getattr(st.context, "theme", None), "type", "light")).lower()
    is_dark = theme_type == "dark"
    embedding_heatmap_colorscale = "magma" if is_dark else "RdBu"
    
    if hasattr(embeddings, "detach"):
        E = embeddings.detach().float().cpu().numpy()
    else:
        E = np.asarray(embeddings, dtype=np.float32)

    assert E.ndim == 3, "Expected shape (num_sequences, num_residues, embedding_dim)"
    N, R, D = E.shape

    # Auto-determine max sequences to plot
    if max_plot_sequences is None:
        max_plot_sequences = min(N, 10)

    N_plot = min(N, max_plot_sequences)
    E_plot = E[:N_plot]
    ids_plot = list(ids[:N_plot])
    residues_plot = list(residues[:N_plot])

    seq_colors = list(plotly.colors.qualitative.Plotly)
    while len(seq_colors) < N_plot:
        seq_colors.extend(plotly.colors.qualitative.Alphabet)

    def _wrapped_labels(labels, max_len=20):
        return [label.replace(" ", "<br>") if len(str(label)) > max_len else str(label) for label in labels]

    results = {
        "mode": mode,
        "shape": (N, R, D),
        "n_sequences_plotted": N_plot,
    }

    # Sequence-level summary
    seq_mean_norm = np.linalg.norm(E_plot.mean(axis=1), axis=1)
    seq_spread = E_plot.std(axis=1).mean(axis=1)
    seq_summary = pd.DataFrame({
        "sequence": ids_plot,
        "mean_norm": seq_mean_norm,
        "mean_spread": seq_spread,
    })

    fig_summary = go.Figure(
        data=go.Scatter(
            x=seq_summary["mean_norm"],
            y=seq_summary["mean_spread"],
            mode="markers+text",
            text=seq_summary["sequence"],
            textposition="top center",
            marker=dict(
                size=10,
                opacity=0.85,
                color=list(range(N_plot)),
            ),
            hovertemplate=(
                "<b>Sequence:</b> %{text}<br>"
                "<b>Mean norm:</b> %{x:.4f}<br>"
                "<b>Mean spread:</b> %{y:.4f}<extra></extra>"
            ),
        )
    )
    fig_summary.update_layout(
        title="Sequence embedding summary: mean norm vs spread",
        xaxis_title="Mean norm",
        yaxis_title="Mean spread",
        width=1200,
        height=500,
    )
    _apply_transparent_background(fig_summary)
    st.plotly_chart(fig_summary, width='stretch')
    results["sequence_summary"] = seq_summary

    if mode == "pca":
        X = E_plot.reshape(N_plot * R, D)
        Xz = StandardScaler().fit_transform(X)
        n_components = min(n_pcs, D, Xz.shape[0])
        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(Xz).reshape(N_plot, R, n_components)

        pc_names = [f"pc{i+1}_score" for i in range(n_components)]

        long_df = pd.DataFrame({
            "sequence_original_idx": np.repeat(np.arange(N_plot), R),
            "residue_original_idx": np.tile(np.arange(R), N_plot),
        })
        for i, pc_name in enumerate(pc_names):
            long_df[pc_name] = pcs[:, :, i].reshape(-1)

        long_df["sequence"] = long_df["sequence_original_idx"].apply(lambda idx: ids_plot[idx])
        long_df["residue"] = long_df.apply(
            lambda row: residues_plot[row["sequence_original_idx"]][row["residue_original_idx"]],
            axis=1,
        )
        long_df = long_df.drop(columns=["sequence_original_idx", "residue_original_idx"])
        long_df = long_df[["sequence", "residue"] + pc_names]

        explained_df = pd.DataFrame({
            "component": [f"PC{i+1}" for i in range(n_components)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        })

        st.markdown("**PCA Explained Variance**")
        st.dataframe(explained_df, width='stretch')

        custom_hover_residue_chars = np.array([
            [residues_plot[seq][res] for res in range(R)]
            for seq in range(N_plot)
        ])

        for comp_idx, pc_name in enumerate(pc_names):
            scores = pcs[:, :, comp_idx]

            fig = go.Figure()
            for seq in range(N_plot):
                for res in range(R):
                    fig.add_trace(
                        go.Box(
                            x=[f"Res {res}"],
                            y=[scores[seq, res]],
                            name=f"Seq {ids_plot[seq]}",
                            legendgroup=f"{ids_plot[seq]}",
                            showlegend=(res == 0),
                            boxpoints="all",
                            jitter=0.25,
                            pointpos=0,
                            marker=dict(
                                size=5,
                                opacity=0.75,
                                color=seq_colors[seq % len(seq_colors)],
                            ),
                            line=dict(width=1),
                            hovertemplate=(
                                f"Sequence {ids_plot[seq]}<br>"
                                f"Residue {res}:{residues_plot[seq][res]}<br>"
                                f"{pc_name.upper()}=%{{y:.4f}}<extra></extra>"
                            ),
                        )
                    )

            fig.update_layout(
                title=f"Residue embedding {pc_name.upper()} score per residue",
                boxmode="group",
                xaxis_title="Residue",
                yaxis_title=pc_name.upper(),
                legend_title="Sequence",
                width=1200,
                legend=dict(orientation="h", yanchor="bottom", y=-0.7, xanchor="right", x=1),
            )
            _apply_transparent_background(fig)
            st.plotly_chart(fig, width='stretch')

            # Heatmap
            heatmap_df = pd.DataFrame(
                scores,
                index=[str(ids_plot[i]) for i in range(N_plot)],
                columns=[f"Res {j}" for j in range(R)],
            )

            fig2 = go.Figure(
                data=go.Heatmap(
                    z=scores,
                    x=[f"Res {i}" for i in range(R)],
                    y=[str(ids_plot[i]) for i in range(N_plot)],
                    colorscale=embedding_heatmap_colorscale,
                    zmid=0,
                    colorbar=dict(orientation="h", y=-0.4),
                    customdata=custom_hover_residue_chars,
                    hovertemplate=(
                        "<b>Sequence:</b> %{y}<br>"
                        "<b>Residue:</b> %{customdata} (%{x})<br>"
                        f"<b>{pc_name.upper()}:</b> %{{z:.4f}}<extra></extra>"
                    ),
                )
            )
            fig2.update_layout(
                title=f"{pc_name.upper()} score heatmap by residue and sequence",
                xaxis_title="Residue",
                yaxis=dict(
                    title="Sequence",
                    automargin=True,
                    tickfont=dict(size=11),
                    categoryorder="trace",
                ),
                width=1200,
                height=800,
                margin=dict(l=50),
            )
            _apply_transparent_background(fig2)
            fig2.update_yaxes(
                ticktext=_wrapped_labels(ids_plot),
                tickvals=[str(ids_plot[i]) for i in range(N_plot)],
            )
            st.plotly_chart(fig2, width='stretch')

        results["pcs"] = pcs
        results["long_df"] = long_df
        results["explained_variance"] = explained_df

    elif mode == "raw_dims":
        mean_over_dims = E_plot.mean(axis=2)

        fig = go.Figure()
        for seq in range(N_plot):
            for res in range(R):
                fig.add_trace(
                    go.Box(
                        x=[f"Res {res}"] * D,
                        y=E_plot[seq, res, :],
                        name=f"Seq {ids_plot[seq]}",
                        legendgroup=f"{ids_plot[seq]}",
                        showlegend=(res == 0),
                        boxpoints="all",
                        jitter=0.25,
                        marker=dict(opacity=0.75, color=seq_colors[seq % len(seq_colors)]),
                        line=dict(width=1),
                        hovertemplate=(
                            f"Sequence {ids_plot[seq]}<br>"
                            f"Residue {res}:{residues_plot[seq][res]}<br>"
                            "Value=%{y:.4f}<extra></extra>"
                        ),
                    )
                )

        fig.update_layout(
            title="Raw embedding dimension distribution per residue",
            boxmode="group",
            xaxis_title="Residue",
            yaxis_title="Embedding value",
            legend_title="Sequence",
            width=1200,
            legend=dict(orientation="h", yanchor="bottom", y=-0.6, xanchor="right", x=1),
        )
        _apply_transparent_background(fig)
        st.plotly_chart(fig, width='stretch')

        # Mean heatmap
        heatmap_df = pd.DataFrame(
            mean_over_dims,
            index=[str(ids_plot[i]) for i in range(N_plot)],
            columns=[f"Res {j}" for j in range(R)],
        )

        custom_hover_residue_chars = np.array([
            [residues_plot[seq][res] for res in range(R)]
            for seq in range(N_plot)
        ])

        fig2 = go.Figure(
            data=go.Heatmap(
                z=mean_over_dims,
                x=[f"Res {i}" for i in range(R)],
                y=[str(ids_plot[i]) for i in range(N_plot)],
                colorscale=embedding_heatmap_colorscale,
                zmid=0,
                colorbar=dict(orientation="h", y=-0.4),
                customdata=custom_hover_residue_chars,
                hovertemplate=(
                    "<b>Sequence:</b> %{y}<br>"
                    "<b>Residue:</b> %{customdata} (%{x})<br>"
                    "<b>Mean over dims:</b> %{z:.4f}<extra></extra>"
                ),
            )
        )
        fig2.update_layout(
            title="Raw embedding mean heatmap by sequence and residue",
            xaxis_title="Residue",
            yaxis=dict(
                title="Sequence",
                automargin=True,
                tickfont=dict(size=11),
                categoryorder="trace",
            ),
            width=1200,
            height=800,
            margin=dict(l=50),
        )
        _apply_transparent_background(fig2)
        fig2.update_yaxes(
            ticktext=_wrapped_labels(ids_plot),
            tickvals=[str(ids_plot[i]) for i in range(N_plot)],
        )
        st.plotly_chart(fig2, width='stretch')
        results["raw_df"] = None

    else:
        raise ValueError("mode must be either 'pca' or 'raw_dims'")

    return results
