"""
app.py  —  Depth Estimation Research Dashboard
All 3 models: CNN · Hybrid · DPT
Pages: Individual + 3-Way Compare + Architecture + Analysis

Run:  python -m streamlit run app.py
"""

import io, sys, os, math
import torch
import numpy as np
import streamlit as st
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cnn_model                    import CNNDepthModel
from hybrid_model                 import HybridDepthModel
from depth_prediction_transformer import DepthPredictionTransformer

# ══════════════════════════════════════════════════════════
# PAGE CONFIG + GLOBAL STYLE
# ══════════════════════════════════════════════════════════

st.set_page_config(page_title="DepthLab", page_icon="🔬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@300;400&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    border: 1px solid #00d4ff22;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 40%, #00d4ff08 0%, transparent 50%),
                radial-gradient(circle at 70% 60%, #7c3aed08 0%, transparent 50%);
}
.main-header h1 {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(90deg, #00d4ff, #7c3aed, #f59e0b);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; letter-spacing: -1px;
}
.main-header p { color: #94a3b8; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; margin: 0.5rem 0 0; }

.model-card {
    border-radius: 12px; padding: 1.2rem 1.5rem;
    border: 1px solid; margin-bottom: 0.8rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
}
.card-cnn   { background: #0f1923; border-color: #00d4ff44; }
.card-hybrid{ background: #130f23; border-color: #7c3aed44; }
.card-dpt   { background: #1a130f; border-color: #f59e0b44; }

.metric-box {
    background: #0f0f1a; border: 1px solid #ffffff11;
    border-radius: 10px; padding: 1rem; text-align: center;
}
.metric-val { font-size: 1.6rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
.metric-label { font-size: 0.7rem; color: #64748b; letter-spacing: 1px; text-transform: uppercase; margin-top: 4px; }

.arch-stage {
    background: linear-gradient(135deg, #0f1923, #0a0f1e);
    border: 1px solid #00d4ff33;
    border-radius: 10px; padding: 1rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
}
.tag {
    display: inline-block; border-radius: 4px;
    padding: 2px 8px; font-size: 0.65rem; font-weight: 700;
    letter-spacing: 1px; text-transform: uppercase; margin: 2px;
}
.tag-blue  { background: #00d4ff22; color: #00d4ff; border: 1px solid #00d4ff44; }
.tag-purple{ background: #7c3aed22; color: #a78bfa; border: 1px solid #7c3aed44; }
.tag-amber { background: #f59e0b22; color: #fbbf24; border: 1px solid #f59e0b44; }
.tag-green { background: #10b98122; color: #34d399; border: 1px solid #10b98144; }

.stMetric { background: #0f1923 !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_H, TRAIN_W = 128, 416

TRANSFORM = transforms.Compose([
    transforms.Resize((TRAIN_H, TRAIN_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

MODEL_REGISTRY = {
    "CNN":    CNNDepthModel,
    "Hybrid": HybridDepthModel,
    "DPT":    DepthPredictionTransformer,
}
MODEL_KWARGS = {
    "CNN":    dict(max_depth=80, pretrained=False),
    "Hybrid": dict(max_depth=80, pretrained_cnn=False,
                   vit_embed_dim=256, vit_num_heads=8,
                   vit_num_layers=4, vit_spatial=(4,13)),
    "DPT":    dict(config='small', patch_size=16, max_depth=80),
}
MODEL_COLORS = {"CNN": "#00d4ff", "Hybrid": "#a78bfa", "DPT": "#fbbf24"}
MODEL_FILES  = {"CNN": "best_cnn_model.pth",
                "Hybrid": "best_hybrid_model.pth",
                "DPT": "best_dpt_model.pth"}

CMAPS = ["plasma","magma","inferno","viridis","turbo","jet","RdYlBu_r"]

# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════

@st.cache_resource
def load_model(name, pth_bytes):
    m = MODEL_REGISTRY[name](**MODEL_KWARGS[name])
    m.load_state_dict(torch.load(io.BytesIO(pth_bytes), map_location=DEVICE))
    return m.to(DEVICE).eval()

def infer(model, pil_img):
    ow, oh = pil_img.size
    t = TRANSFORM(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        d = model(t)
    return F.interpolate(d,(oh,ow),mode="bilinear",align_corners=True).squeeze().cpu().numpy()

def colorize(d, cmap="plasma"):
    n = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return Image.fromarray((cm.get_cmap(cmap)(n)[:,:,:3]*255).astype(np.uint8))

def get_image(prefix):
    src = st.radio("Input", ["📁 Upload","📷 Webcam"], horizontal=True, key=f"{prefix}_src")
    img = None
    if src == "📁 Upload":
        f = st.file_uploader("Image", type=["png","jpg","jpeg","bmp"], key=f"{prefix}_up")
        if f: img = Image.open(f)
    else:
        c = st.camera_input("Capture", key=f"{prefix}_cam")
        if c: img = Image.open(c)
    return img

def depth_stats(d):
    valid = d[d > 0] if d.min() >= 0 else d.flatten()
    return {"min": float(d.min()), "max": float(d.max()),
            "mean": float(d.mean()), "std": float(d.std()),
            "near_pct": float((d < d.mean()).mean() * 100)}

def show_single_result(img, d_np, cmap, model_name):
    col = MODEL_COLORS[model_name]
    d_pil = colorize(d_np, cmap)
    c1, c2 = st.columns(2)
    with c1:
        st.image(np.array(img.convert("RGB")), use_container_width=True,
                 caption="Input Image")
    with c2:
        st.image(d_pil, use_container_width=True,
                 caption=f"{model_name} Depth Map")
    st.divider()

    # Metrics row
    stats = depth_stats(d_np)
    cols  = st.columns(5)
    labels = ["Min (m)","Max (m)","Mean (m)","Std Dev","% Near"]
    vals   = [f"{stats['min']:.2f}", f"{stats['max']:.2f}",
              f"{stats['mean']:.2f}", f"{stats['std']:.2f}",
              f"{stats['near_pct']:.1f}%"]
    for c, l, v in zip(cols, labels, vals):
        c.markdown(f"""<div class="metric-box">
            <div class="metric-val" style="color:{col}">{v}</div>
            <div class="metric-label">{l}</div></div>""", unsafe_allow_html=True)

    st.divider()

    # Multi-panel analysis
    st.markdown("#### 📊 Depth Analysis")
    fig = plt.figure(figsize=(18, 10), facecolor='#0a0a14')
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)

    rgb_np = np.array(img.convert("RGB").resize((d_np.shape[1], d_np.shape[0])))

    # 1. RGB
    ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(rgb_np); ax1.set_title("RGB Input", color='white', fontsize=10); ax1.axis('off')

    # 2. Depth heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(d_np, cmap=cmap); ax2.set_title("Depth Map", color='white', fontsize=10); ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white', labelcolor='white')

    # 3. Depth histogram
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor('#0f0f1a')
    vals_flat = d_np.flatten()
    ax3.hist(vals_flat[vals_flat > 0], bins=60, color=col, alpha=0.8, edgecolor='none')
    ax3.axvline(stats['mean'], color='white', linestyle='--', linewidth=1.5, label=f"Mean: {stats['mean']:.1f}m")
    ax3.set_title("Depth Distribution", color='white', fontsize=10)
    ax3.set_xlabel("Depth (m)", color='#94a3b8', fontsize=8)
    ax3.set_ylabel("Pixel Count", color='#94a3b8', fontsize=8)
    ax3.tick_params(colors='#64748b'); ax3.legend(fontsize=7, labelcolor='white')
    for sp in ax3.spines.values(): sp.set_edgecolor('#1e293b')

    # 4. Depth gradient (edges)
    ax4 = fig.add_subplot(gs[0, 3])
    gy, gx = np.gradient(d_np); grad_mag = np.sqrt(gx**2 + gy**2)
    ax4.imshow(grad_mag, cmap='hot'); ax4.set_title("Depth Gradient (Edges)", color='white', fontsize=10); ax4.axis('off')

    # 5. Near / mid / far segmentation
    ax5 = fig.add_subplot(gs[1, 0])
    zones = np.zeros((*d_np.shape, 3), dtype=np.uint8)
    near_th = stats['mean'] * 0.6; far_th = stats['mean'] * 1.4
    zones[d_np < near_th]  = [0, 200, 255]
    zones[(d_np >= near_th) & (d_np < far_th)] = [120, 80, 200]
    zones[d_np >= far_th]  = [255, 160, 0]
    ax5.imshow(zones); ax5.set_title("Near/Mid/Far Zones", color='white', fontsize=10); ax5.axis('off')
    for label, c_ in [("Near", (0,200,255)), ("Mid", (120,80,200)), ("Far", (255,160,0))]:
        ax5.plot([], [], 's', color=[x/255 for x in c_], label=label, markersize=8)
    ax5.legend(loc='lower right', fontsize=7, labelcolor='white', framealpha=0.4)

    # 6. Depth rows (horizontal slices)
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.set_facecolor('#0f0f1a')
    h = d_np.shape[0]
    for i, row_frac in enumerate([0.2, 0.5, 0.8]):
        row = int(h * row_frac)
        ax6.plot(d_np[row], linewidth=1.2, alpha=0.9,
                 label=f"Row {int(row_frac*100)}%")
    ax6.set_title("Horizontal Depth Profiles", color='white', fontsize=10)
    ax6.set_xlabel("Column pixel", color='#94a3b8', fontsize=8)
    ax6.set_ylabel("Depth (m)", color='#94a3b8', fontsize=8)
    ax6.tick_params(colors='#64748b'); ax6.legend(fontsize=7, labelcolor='white')
    for sp in ax6.spines.values(): sp.set_edgecolor('#1e293b')

    # 7. Depth columns (vertical slices)
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.set_facecolor('#0f0f1a')
    w = d_np.shape[1]
    for i, col_frac in enumerate([0.2, 0.5, 0.8]):
        col_i = int(w * col_frac)
        ax7.plot(d_np[:, col_i], range(h), linewidth=1.2, alpha=0.9,
                 label=f"Col {int(col_frac*100)}%")
    ax7.invert_yaxis()
    ax7.set_title("Vertical Depth Profiles", color='white', fontsize=10)
    ax7.set_xlabel("Depth (m)", color='#94a3b8', fontsize=8)
    ax7.set_ylabel("Row pixel", color='#94a3b8', fontsize=8)
    ax7.tick_params(colors='#64748b'); ax7.legend(fontsize=7, labelcolor='white')
    for sp in ax7.spines.values(): sp.set_edgecolor('#1e293b')

    # 8. Depth heatmap overlay on RGB
    ax8 = fig.add_subplot(gs[1, 3])
    norm_d = (d_np - d_np.min()) / (d_np.max() - d_np.min() + 1e-8)
    heat   = (cm.get_cmap(cmap)(norm_d)[:,:,:3] * 255).astype(np.uint8)
    blend  = (rgb_np.astype(float) * 0.45 + heat.astype(float) * 0.55).clip(0,255).astype(np.uint8)
    ax8.imshow(blend); ax8.set_title("RGB + Depth Overlay", color='white', fontsize=10); ax8.axis('off')

    for ax in [ax1,ax2,ax4,ax5,ax8]:
        ax.set_facecolor('#0a0a14')
    fig.patch.set_facecolor('#0a0a14')

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Download
    buf = io.BytesIO(); d_pil.save(buf, "PNG")
    st.download_button("⬇️ Download Depth Map", buf.getvalue(), "depth.png", "image/png")


# ══════════════════════════════════════════════════════════
# ARCHITECTURE DIAGRAM (matplotlib)
# ══════════════════════════════════════════════════════════

def draw_architecture_diagram(model_name):
    fig, ax = plt.subplots(1, 1, figsize=(16, 5), facecolor='#0a0a14')
    ax.set_facecolor('#0a0a14'); ax.axis('off')
    ax.set_xlim(0, 16); ax.set_ylim(0, 5)

    col = MODEL_COLORS[model_name]

    def box(x, y, w, h, label, sublabel="", color="#00d4ff", alpha=0.15):
        rect = plt.Rectangle((x, y), w, h, linewidth=1.5,
                               edgecolor=color, facecolor=color,
                               alpha=alpha, zorder=2)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2+(0.15 if sublabel else 0), label,
                ha='center', va='center', color=color,
                fontsize=8, fontweight='bold', zorder=3)
        if sublabel:
            ax.text(x+w/2, y+h/2-0.22, sublabel, ha='center', va='center',
                    color=color, fontsize=6.5, alpha=0.8, zorder=3,
                    fontfamily='monospace')

    def arrow(x1, y1, x2, y2, color="#ffffff"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color,
                                   lw=1.5, connectionstyle="arc3,rad=0"))

    cy = 2.1  # centre y

    if model_name == "CNN":
        items = [
            (0.2, "Input\nRGB", "[B,3,H,W]", "#64748b"),
            (1.8, "enc1\nVGG", "[64,H/2]",  "#00d4ff"),
            (3.4, "enc2\nVGG", "[128,H/4]", "#00d4ff"),
            (5.0, "enc3\nVGG", "[256,H/8]", "#0ea5e9"),
            (6.6, "enc4\nVGG", "[512,H/16]","#0ea5e9"),
            (8.2, "enc5\nVGG", "[512,H/32]","#0284c7"),
            (9.8, "up5+\nskip4","[256,H/16]","#7dd3fc"),
            (11.2,"up4+\nskip3","[128,H/8]", "#bae6fd"),
            (12.6,"up3+\nskip2","[64,H/4]",  "#e0f2fe"),
            (14.0,"Depth\nHead", "[1,H,W]",  "#f8fafc"),
        ]
        for i,(x,lbl,sub,c) in enumerate(items):
            box(x, cy-0.7, 1.3, 1.4, lbl, sub, c)
            if i < len(items)-1:
                arrow(x+1.3, cy, items[i+1][0], cy, "#ffffff44")

        # Skip connection arcs
        for (xa, xb, yt) in [(8.2,14.0,4.2),(6.6,11.2,4.6),(5.0,12.6,4.0)]:
            ax.annotate("", xy=(xb+0.65, cy+0.7), xytext=(xa+0.65, cy+0.7),
                        arrowprops=dict(arrowstyle="->", color="#00d4ff55",
                                       lw=1, connectionstyle=f"arc3,rad=-0.3"))
        ax.text(8, 4.5, "skip connections", color="#00d4ff55",
                fontsize=7, ha='center', style='italic')

    elif model_name == "Hybrid":
        items = [
            (0.2,  "Input\nRGB",     "[B,3,H,W]",   "#64748b"),
            (1.8,  "CNN\nEncoder",   "VGG16 ×5",    "#00d4ff"),
            (4.2,  "enc5",           "[512,H/32]",  "#0284c7"),
            (5.8,  "ViT\nBottleneck","52 tokens",   "#a78bfa"),
            (7.6,  "4× Transformer\nBlocks","attn+FFN","#7c3aed"),
            (9.6,  "Enriched\nfeats", "[512,H/32]", "#a78bfa"),
            (11.2, "CNN\nDecoder",   "U-Net ×5",    "#00d4ff"),
            (13.4, "Depth\nHead",    "[1,H,W]",     "#f8fafc"),
        ]
        widths = [1.4,2.0,1.2,1.4,1.6,1.4,1.8,1.4]
        xs = [0.2,1.8,4.2,5.8,7.6,9.6,11.2,13.4]
        cols_= ["#64748b","#00d4ff","#0284c7","#a78bfa","#7c3aed","#a78bfa","#00d4ff","#f8fafc"]
        for i,(x,lbl,sub,c) in enumerate(items):
            w = widths[i]
            box(x, cy-0.7, w, 1.4, lbl, sub, c)
            if i < len(items)-1:
                arrow(x+w, cy, xs[i+1], cy, "#ffffff44")

        # Skip arc over ViT
        ax.annotate("", xy=(11.2+0.9, cy+0.7), xytext=(1.8+1.0, cy+0.7),
                    arrowprops=dict(arrowstyle="->", color="#00d4ff44",
                                   lw=1.2, connectionstyle="arc3,rad=-0.25"))
        ax.text(7, 4.6, "5 skip connections bypass ViT bottleneck",
                color="#00d4ff55", fontsize=7, ha='center', style='italic')

        # ViT annotation
        rect2 = plt.Rectangle((5.6, cy-1.0), 5.6, 2.0, linewidth=1,
                               edgecolor="#7c3aed", facecolor="#7c3aed",
                               alpha=0.06, zorder=1, linestyle='--')
        ax.add_patch(rect2)
        ax.text(8.4, cy+1.3, "ViT Bottleneck Stage", color="#a78bfa",
                fontsize=7.5, ha='center', style='italic')

    else:  # DPT
        # Stage 1
        s1_items = [(0.3,"Patch\nEmbed","16×16",  "#fbbf24"),
                    (1.8,"ViT Layers\n1–3",  "tap→s1","#f59e0b"),
                    (3.3,"ViT Layers\n4–6",  "tap→s2","#d97706"),
                    (4.8,"ViT Layers\n7–9",  "tap→s3","#b45309"),
                    (6.3,"ViT Layers\n10–12","tap→s4","#92400e")]
        for x,lbl,sub,c in s1_items:
            box(x, cy+0.6, 1.3, 1.1, lbl, sub, c)
        for i in range(len(s1_items)-1):
            arrow(s1_items[i][0]+1.3, cy+1.15, s1_items[i+1][0], cy+1.15, "#fbbf2466")

        # Reassemble + FPN
        fpn_items = [(1.8,"Reassemble\n×4","4 scales","#fbbf24"),
                     (3.5,"FPN\nFusion","bottom-up","#f59e0b")]
        for x,lbl,sub,c in fpn_items:
            box(x, cy-1.7, 1.5, 1.1, lbl, sub, c)

        # Stage 2 + 3
        stage23 = [(6.0,"Stage 2\nMid ViT","pool 16×16","#a78bfa"),
                   (7.8,"Stage 3\nDec ViT","pool 8×8",  "#7c3aed"),
                   (9.6,"Depth\nHead",     "[1,H,W]",   "#f8fafc")]
        for x,lbl,sub,c in stage23:
            box(x, cy-0.7, 1.5, 1.4, lbl, sub, c)
        for i in range(len(stage23)-1):
            arrow(stage23[i][0]+1.5, cy, stage23[i+1][0], cy, "#ffffff44")

        # Arrows from ViT to reassemble
        for xi in [1.8,3.3,4.8,6.3]:
            ax.annotate("", xy=(2.55, cy-0.6), xytext=(xi+0.65, cy+0.6),
                        arrowprops=dict(arrowstyle="->", color="#fbbf2444",
                                       lw=0.8, connectionstyle="arc3,rad=0.1"))

        # Arrow fpn → stage2
        arrow(5.0, cy-1.15, 6.0, cy-0.0, "#fbbf2466")

        ax.text(0.3, 4.1, "STAGE 1 — Encoder ViT", color="#fbbf2488",
                fontsize=7.5, style='italic')
        ax.text(6.0, 4.1, "STAGE 2&3 — Decoder ViTs",
                color="#a78bfa88", fontsize=7.5, style='italic')

    ax.set_title(f"{model_name} Architecture", color=col,
                 fontsize=13, fontweight='bold', pad=10)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""<div style='text-align:center;padding:1rem 0'>
        <div style='font-size:2rem'>🔬</div>
        <div style='font-size:1.2rem;font-weight:800;background:linear-gradient(90deg,#00d4ff,#a78bfa);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent'>DepthLab</div>
        <div style='font-size:0.65rem;color:#475569;font-family:monospace'>Monocular Depth Research</div>
    </div>""", unsafe_allow_html=True)

    st.divider()
    page = st.radio("", [
        "🏠 Home",
        "🧠 CNN Model",
        "🔀 Hybrid CNN+ViT",
        "⚡ DPT Transformer",
        "📊 3-Way Comparison",
        "🏗️ Architectures",
    ], label_visibility="collapsed")

    st.divider()
    cmap = st.selectbox("Colormap", CMAPS, index=0)
    st.divider()
    st.caption(f"**Device:** `{DEVICE}`")
    st.caption(f"**Input size:** {TRAIN_W}×{TRAIN_H}")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.7rem;color:#475569;font-family:monospace;line-height:1.8'>
    🟡 warm = close<br>🔵 cool = far<br>
    ──────────────<br>
    CNN → best_cnn_model.pth<br>
    Hybrid → best_hybrid_model.pth<br>
    DPT → best_dpt_model.pth
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════

if page == "🏠 Home":
    st.markdown("""<div class="main-header">
        <h1>DepthLab 🔬</h1>
        <p>Monocular Depth Estimation Research Dashboard  •  CNN · Hybrid · DPT</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    cards = [
        ("CNN", "#00d4ff", "VGG16 + U-Net", "Fast & data-efficient", "~20M params", "🧠"),
        ("Hybrid", "#a78bfa", "CNN + ViT Bottleneck", "Local + global context", "~20M params", "🔀"),
        ("DPT", "#fbbf24", "3-Stage ViT", "Sharpest boundaries", "~27M params", "⚡"),
    ]
    for col_st, (name, color, arch, tagline, params, icon) in zip([c1,c2,c3], cards):
        with col_st:
            st.markdown(f"""<div class="model-card card-{'cnn' if name=='CNN' else 'hybrid' if name=='Hybrid' else 'dpt'}">
                <div style='font-size:1.8rem'>{icon}</div>
                <div style='font-size:1.1rem;font-weight:800;color:{color};margin:4px 0'>{name}</div>
                <div style='color:#94a3b8;font-size:0.75rem'>{arch}</div>
                <div style='color:#64748b;font-size:0.7rem;margin-top:6px'>{tagline}</div>
                <div style='margin-top:8px'><span class="tag tag-{'blue' if name=='CNN' else 'purple' if name=='Hybrid' else 'amber'}">{params}</span></div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🗺️ Architecture Overview")

    col_l, col_r = st.columns([3,2])
    with col_l:
        st.markdown("""
| Feature | CNN | Hybrid | DPT |
|---|:---:|:---:|:---:|
| Pretrained backbone | VGG16 ✓ | VGG16 ✓ | ✗ |
| Global attention | ✗ | Bottleneck | 3-stage |
| Skip connections | ✓ | ✓ | FPN |
| Multi-scale features | 5 levels | 5 levels | 4 taps |
| Training speed | ⚡⚡⚡ | ⚡⚡ | ⚡ |
| Edge sharpness | ★★★ | ★★★★ | ★★★★★ |
""")
    with col_r:
        st.markdown("""<div class="arch-stage">
<div style='color:#00d4ff;font-weight:bold;margin-bottom:8px'>📂 Required Files</div>
<div style='color:#94a3b8'>Place all in same folder:</div>
<br>
<div style='color:#fbbf24'>cnn_model.py</div>
<div style='color:#a78bfa'>hybrid_model.py</div>
<div style='color:#fbbf24'>depth_prediction_transformer.py</div>
<div style='color:#64748b'>vit_model.py</div>
<div style='color:#64748b'>find_sequences.py</div>
<br>
<div style='color:#34d399'>best_cnn_model.pth</div>
<div style='color:#34d399'>best_hybrid_model.pth</div>
<div style='color:#34d399'>best_dpt_model.pth</div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE: CNN
# ══════════════════════════════════════════════════════════

elif page == "🧠 CNN Model":
    st.markdown("""<div class="main-header">
        <h1 style='font-size:2rem'>🧠 CNN Depth Model</h1>
        <p>VGG16 Encoder + U-Net Decoder  •  cnn_model.py → CNNDepthModel</p>
    </div>""", unsafe_allow_html=True)

    st.pyplot(draw_architecture_diagram("CNN"), use_container_width=True)
    plt.close()

    st.divider()
    pth = st.file_uploader("Upload `best_cnn_model.pth`", type=["pth"], key="cnn_f")
    if not pth: st.info("👆 Upload your trained CNN weights to begin."); st.stop()

    with st.spinner("Loading CNN…"): model = load_model("CNN", pth.read())
    st.success("✅ CNN ready")
    st.divider()

    img = get_image("cnn")
    if img:
        with st.spinner("Estimating…"):
            d = infer(model, img)
        show_single_result(img, d, cmap, "CNN")


# ══════════════════════════════════════════════════════════
# PAGE: HYBRID
# ══════════════════════════════════════════════════════════

elif page == "🔀 Hybrid CNN+ViT":
    st.markdown("""<div class="main-header">
        <h1 style='font-size:2rem'>🔀 Hybrid CNN + ViT</h1>
        <p>VGG16 Encoder → ViT Bottleneck → U-Net Decoder  •  hybrid_model.py → HybridDepthModel</p>
    </div>""", unsafe_allow_html=True)

    st.pyplot(draw_architecture_diagram("Hybrid"), use_container_width=True)
    plt.close()

    st.divider()
    pth = st.file_uploader("Upload `best_hybrid_model.pth`", type=["pth"], key="hyb_f")
    if not pth: st.info("👆 Upload your trained Hybrid weights to begin."); st.stop()

    with st.spinner("Loading Hybrid…"): model = load_model("Hybrid", pth.read())
    st.success("✅ Hybrid ready")
    st.divider()

    img = get_image("hyb")
    if img:
        with st.spinner("Estimating…"):
            d = infer(model, img)
        show_single_result(img, d, cmap, "Hybrid")


# ══════════════════════════════════════════════════════════
# PAGE: DPT
# ══════════════════════════════════════════════════════════

elif page == "⚡ DPT Transformer":
    st.markdown("""<div class="main-header">
        <h1 style='font-size:2rem'>⚡ Dense Prediction Transformer</h1>
        <p>3-Stage ViT  •  depth_prediction_transformer.py → DepthPredictionTransformer</p>
    </div>""", unsafe_allow_html=True)

    st.pyplot(draw_architecture_diagram("DPT"), use_container_width=True)
    plt.close()

    st.divider()
    pth = st.file_uploader("Upload `best_dpt_model.pth`", type=["pth"], key="dpt_f")
    if not pth: st.info("👆 Upload your trained DPT weights to begin."); st.stop()

    with st.spinner("Loading DPT…"): model = load_model("DPT", pth.read())
    st.success("✅ DPT ready")
    st.divider()

    img = get_image("dpt")
    if img:
        with st.spinner("Estimating…"):
            d = infer(model, img)
        show_single_result(img, d, cmap, "DPT")


# ══════════════════════════════════════════════════════════
# PAGE: 3-WAY COMPARISON
# ══════════════════════════════════════════════════════════

elif page == "📊 3-Way Comparison":
    st.markdown("""<div class="main-header">
        <h1 style='font-size:2rem'>📊 3-Way Model Comparison</h1>
        <p>Run all three models on the same image and compare every dimension</p>
    </div>""", unsafe_allow_html=True)

    # Upload all 3 .pth files
    st.subheader("1. Upload Models")
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        st.markdown('<span class="tag tag-blue">CNN</span>', unsafe_allow_html=True)
        pth_cnn = st.file_uploader("CNN .pth", type=["pth"], key="cmp_cnn")
    with pc2:
        st.markdown('<span class="tag tag-purple">Hybrid</span>', unsafe_allow_html=True)
        pth_hyb = st.file_uploader("Hybrid .pth", type=["pth"], key="cmp_hyb")
    with pc3:
        st.markdown('<span class="tag tag-amber">DPT</span>', unsafe_allow_html=True)
        pth_dpt = st.file_uploader("DPT .pth", type=["pth"], key="cmp_dpt")

    st.divider()
    st.subheader("2. Input Image")
    img = get_image("cmp")

    if not img:
        st.info("👆 Upload an image above.")
        st.stop()

    available = {}
    if pth_cnn: available["CNN"]    = pth_cnn
    if pth_hyb: available["Hybrid"] = pth_hyb
    if pth_dpt: available["DPT"]    = pth_dpt

    if not available:
        st.warning("Upload at least one `.pth` file above.")
        st.stop()

    with st.spinner("Loading models & running inference…"):
        results = {}
        for name, pth_f in available.items():
            m = load_model(name, pth_f.read())
            results[name] = infer(m, img)

    st.divider()
    st.subheader("3. Depth Maps")

    # Show all depth maps side by side
    dep_cols = st.columns(len(results) + 1)
    with dep_cols[0]:
        st.image(np.array(img.convert("RGB")), use_container_width=True,
                 caption="📷 Input")
    for col_st, (name, d) in zip(dep_cols[1:], results.items()):
        with col_st:
            dp = colorize(d, cmap)
            st.image(dp, use_container_width=True, caption=f"{name} Depth")
            buf = io.BytesIO(); dp.save(buf,"PNG")
            st.download_button(f"⬇️ {name}", buf.getvalue(),
                               f"depth_{name.lower()}.png", "image/png")

    st.divider()
    st.subheader("4. Metrics Dashboard")

    # Metrics table
    mc = st.columns(len(results))
    for col_st, (name, d) in zip(mc, results.items()):
        color = MODEL_COLORS[name]
        s = depth_stats(d)
        with col_st:
            st.markdown(f"<div style='color:{color};font-weight:800;font-size:1rem;margin-bottom:8px'>{name}</div>", unsafe_allow_html=True)
            for label, val in [("Min", f"{s['min']:.2f} m"),
                                ("Max", f"{s['max']:.2f} m"),
                                ("Mean", f"{s['mean']:.2f} m"),
                                ("Std Dev", f"{s['std']:.2f} m"),
                                ("% Near", f"{s['near_pct']:.1f}%")]:
                st.markdown(f"""<div class="metric-box" style="margin-bottom:6px">
                    <div class="metric-val" style="color:{color};font-size:1.2rem">{val}</div>
                    <div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("5. Comparative Analysis Charts")

    fig = plt.figure(figsize=(18, 14), facecolor='#0a0a14')
    gs  = gridspec.GridSpec(3, len(results)+1, figure=fig,
                            hspace=0.4, wspace=0.3)

    rgb_np = np.array(img.convert("RGB").resize((list(results.values())[0].shape[1],
                                                  list(results.values())[0].shape[0])))

    # Row 0: RGB + depth maps with gradient overlay
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgb_np); ax.set_title("RGB Input", color='white', fontsize=10); ax.axis('off')
    for j,(name,d) in enumerate(results.items()):
        color = MODEL_COLORS[name]
        ax = fig.add_subplot(gs[0, j+1])
        im = ax.imshow(d, cmap=cmap, vmin=0, vmax=80)
        ax.set_title(f"{name}", color=color, fontsize=10, fontweight='bold'); ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white', labelcolor='white')

    # Row 1: Depth histograms
    ax0 = fig.add_subplot(gs[1, 0]); ax0.set_facecolor('#0f0f1a'); ax0.axis('off')
    ax0.text(0.5, 0.5, "Depth\nHistograms", ha='center', va='center',
             color='white', fontsize=11, fontweight='bold', transform=ax0.transAxes)
    for j,(name,d) in enumerate(results.items()):
        color = MODEL_COLORS[name]
        ax = fig.add_subplot(gs[1, j+1]); ax.set_facecolor('#0f0f1a')
        vals = d.flatten(); vals = vals[vals > 0]
        ax.hist(vals, bins=50, color=color, alpha=0.75, edgecolor='none')
        ax.axvline(vals.mean(), color='white', linestyle='--', lw=1.5,
                   label=f"μ={vals.mean():.1f}m")
        ax.set_xlabel("Depth (m)", color='#94a3b8', fontsize=8)
        ax.set_ylabel("Count", color='#94a3b8', fontsize=8)
        ax.tick_params(colors='#64748b'); ax.legend(fontsize=7, labelcolor='white')
        for sp in ax.spines.values(): sp.set_edgecolor('#1e293b')

    # Row 2: Difference maps (only if multiple models)
    if len(results) >= 2:
        names_list = list(results.keys())
        diffs_info = []
        for i in range(len(names_list)):
            for j in range(i+1, len(names_list)):
                diffs_info.append((names_list[i], names_list[j]))

        ax0 = fig.add_subplot(gs[2, 0]); ax0.set_facecolor('#0f0f1a'); ax0.axis('off')
        ax0.text(0.5, 0.5, "Difference\nMaps", ha='center', va='center',
                 color='white', fontsize=11, fontweight='bold', transform=ax0.transAxes)

        for k,(na,nb) in enumerate(diffs_info[:len(results)]):
            diff = np.abs(results[na] - results[nb])
            ax = fig.add_subplot(gs[2, k+1])
            im = ax.imshow(diff, cmap='hot', vmin=0, vmax=20)
            ax.set_title(f"|{na}−{nb}|  μ={diff.mean():.2f}m",
                         color='white', fontsize=9); ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white', labelcolor='white')
    else:
        # Single model: show gradient magnitude
        ax0 = fig.add_subplot(gs[2, 0]); ax0.set_facecolor('#0f0f1a'); ax0.axis('off')
        for j,(name,d) in enumerate(results.items()):
            gy,gx = np.gradient(d); gm = np.sqrt(gx**2+gy**2)
            ax = fig.add_subplot(gs[2,j+1])
            ax.imshow(gm, cmap='hot'); ax.set_title(f"{name} Gradient",color='white',fontsize=9); ax.axis('off')

    for ax_i in fig.get_axes():
        ax_i.set_facecolor('#0a0a14')
    fig.patch.set_facecolor('#0a0a14')
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()
    st.subheader("6. Depth Profile Comparison")

    # Overlay horizontal profiles for all models
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 4), facecolor='#0a0a14')
    h_ref = list(results.values())[0].shape[0]
    w_ref = list(results.values())[0].shape[1]

    for ax_i, row_frac, title in zip(axes2, [0.25, 0.5, 0.75],
                                     ["Top 25%", "Middle 50%", "Bottom 75%"]):
        ax_i.set_facecolor('#0f0f1a')
        row = int(h_ref * row_frac)
        for name, d in results.items():
            ax_i.plot(d[row], label=name, color=MODEL_COLORS[name],
                      linewidth=1.8, alpha=0.9)
        ax_i.set_title(f"Horizontal Slice — {title}", color='white', fontsize=10)
        ax_i.set_xlabel("Column pixel", color='#94a3b8', fontsize=8)
        ax_i.set_ylabel("Depth (m)", color='#94a3b8', fontsize=8)
        ax_i.tick_params(colors='#64748b')
        ax_i.legend(fontsize=8, labelcolor='white', framealpha=0.3)
        for sp in ax_i.spines.values(): sp.set_edgecolor('#1e293b')

    fig2.patch.set_facecolor('#0a0a14')
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)


# ══════════════════════════════════════════════════════════
# PAGE: ARCHITECTURES
# ══════════════════════════════════════════════════════════

elif page == "🏗️ Architectures":
    st.markdown("""<div class="main-header">
        <h1 style='font-size:2rem'>🏗️ Architecture Reference</h1>
        <p>Visual breakdown of all three model architectures</p>
    </div>""", unsafe_allow_html=True)

    for name in ["CNN", "Hybrid", "DPT"]:
        color = MODEL_COLORS[name]
        st.markdown(f"### <span style='color:{color}'>{name}</span>", unsafe_allow_html=True)
        st.pyplot(draw_architecture_diagram(name), use_container_width=True)
        plt.close()

        if name == "CNN":
            st.markdown("""<div class="model-card card-cnn">
<b>Key design choices:</b><br>
• VGG16 pretrained on ImageNet — strong local feature extractor out of the box<br>
• 5 encoder blocks, each halving spatial resolution and doubling channels<br>
• U-Net decoder — each level upsamples ×2 and concatenates the skip connection from the matching encoder level<br>
• enc1 + enc2 frozen — no need to retrain basic edges/colours<br>
• BerHu + SSIM + Edge gradient loss — pixel accuracy + structure + sharp edges
</div>""", unsafe_allow_html=True)

        elif name == "Hybrid":
            st.markdown("""<div class="model-card card-hybrid">
<b>Key design choices:</b><br>
• Same VGG16 encoder as CNN — preserves pretrained features<br>
• ViT bottleneck inserted at enc5 (4×13 = 52 tokens) — global attention is trivially cheap here<br>
• 4 transformer blocks with embed_dim=256, 8 heads — balanced capacity vs compute<br>
• Decoder unchanged from CNN — skip connections bypass the ViT bottleneck entirely<br>
• Result: CNN's local precision + ViT's global scene understanding
</div>""", unsafe_allow_html=True)

        else:
            st.markdown("""<div class="model-card card-dpt">
<b>Key design choices:</b><br>
• Stage 1 (12-layer ViT): full attention on image patches — captures global layout<br>
• 4 tap points at layers 3,6,9,12 → reassemble into multi-scale 2D maps → FPN fusion<br>
• Stage 2 (4-layer ViT): pooled attention (16×16=256 tokens) on encoder output — depth reasoning<br>
• Stage 3 (2-layer ViT): pooled attention (8×8=64 tokens) on mid-resolution — boundary refinement<br>
• Pooled attention: pool to fixed grid → attend → upsample residual (170× memory saving vs full attention)
</div>""", unsafe_allow_html=True)

        st.divider()