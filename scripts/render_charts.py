import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl

# Premium Dark Mode Theme
BG_COLOR = "#0D1117" # GitHub Dark
PANEL_COLOR = "#161B22"
TEXT_COLOR = "#E6EDF3"
ACCENT_GREEN = "#238636"
ACCENT_BLUE = "#58A6FF"
ACCENT_PURPLE = "#8957E5"
RED_WARNING = "#F85149"
GRID_COLOR = "#30363D"

mpl.rcParams['text.color'] = TEXT_COLOR
mpl.rcParams['axes.labelcolor'] = TEXT_COLOR
mpl.rcParams['xtick.color'] = TEXT_COLOR
mpl.rcParams['ytick.color'] = TEXT_COLOR
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Inter', 'Roboto', 'Arial']

def create_bar_chart(lang="en"):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    formats = ['NRA', 'ZIP', 'Tar.gz']
    packing_times = [3.3, 13.4, 38.0]
    
    y_pos = np.arange(len(formats))
    
    # Glowing effect
    for i, pack_time in enumerate(packing_times):
        color = ACCENT_PURPLE if i == 0 else GRID_COLOR
        # Glow
        ax.barh(y_pos[i], pack_time, color=color, alpha=0.3, height=0.6, linewidth=0)
        # Main bar
        ax.barh(y_pos[i], pack_time, color=color, height=0.4, edgecolor=ACCENT_BLUE if i==0 else 'none', linewidth=1)
        
        ax.text(pack_time + 0.5, y_pos[i], f"{pack_time}s", va='center', color=TEXT_COLOR, fontweight='bold')

    ax.set_yticks(y_pos)
    labels = ax.set_yticklabels([f"{f}" for i, f in enumerate(formats)], fontweight='bold', fontsize=12)
    for i, label in enumerate(labels):
        if formats[i] == 'NRA':
            label.set_color(ACCENT_GREEN)
        else:
            label.set_color(RED_WARNING)
    ax.invert_yaxis()  # NRA at top
    
    title = 'Packing Time (60,000 files)' if lang == "en" else 'Время упаковки (60,000 файлов)'
    ax.set_title(title, pad=20, fontsize=16, fontweight='bold', color=TEXT_COLOR)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.grid(axis='x', color=GRID_COLOR, linestyle='--', alpha=0.7)
    
    suffix = "" if lang == "en" else "_ru"
    out_path = f"../docs/assets/archiver_benchmark{suffix}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR, bbox_inches='tight', transparent=False)
    plt.close()

def create_radar_chart(lang="en"):
    categories = ['Cloud Streaming', 'Random Access', 'PyTorch Native', 'Deduplication', 'Encryption', 'Fault Tolerance']
    if lang == "ru":
        categories = ['Cloud Streaming', 'Случайный доступ', 'PyTorch Native', 'Дедупликация', 'Шифрование', 'Отказоустойчивость']
        
    N = len(categories)
    
    # Values [0-10]
    nra_vals = [10, 10, 10, 9, 10, 9]
    tar_vals = [0, 1, 3, 0, 0, 5]
    webdataset = [8, 2, 9, 0, 0, 6]
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    nra_vals += nra_vals[:1]
    tar_vals += tar_vals[:1]
    webdataset += webdataset[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    # Draw NRA
    ax.plot(angles, nra_vals, linewidth=2, linestyle='solid', color=ACCENT_BLUE, label='NRA v4.5')
    ax.fill(angles, nra_vals, color=ACCENT_PURPLE, alpha=0.4)
    
    # Draw WebDataset
    ax.plot(angles, webdataset, linewidth=1.5, linestyle='dashed', color='#D29922', label='WebDataset')
    
    # Draw Tar.gz
    ax.plot(angles, tar_vals, linewidth=1.5, linestyle='dotted', color=GRID_COLOR, label='Tar.gz')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color=TEXT_COLOR, size=11)
    
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels([])
    
    ax.spines['polar'].set_color(GRID_COLOR)
    ax.grid(color=GRID_COLOR, linestyle='--')
    
    title = 'NRA vs Legacy Formats' if lang == "en" else 'NRA против устаревших форматов'
    plt.title(title, size=16, color=TEXT_COLOR, y=1.1, fontweight='bold')
    
    # Legend
    legend = plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), facecolor=PANEL_COLOR, edgecolor=GRID_COLOR)
    texts = legend.get_texts()
    texts[0].set_color(ACCENT_GREEN) # NRA
    texts[1].set_color('#D29922') # WebDataset
    texts[2].set_color(RED_WARNING) # Tar.gz
        
    suffix = "" if lang == "en" else "_ru"
    out_path = f"../docs/assets/radar{suffix}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR, bbox_inches='tight', transparent=False)
    plt.close()

if __name__ == "__main__":
    os.makedirs("../docs/assets", exist_ok=True)
    create_bar_chart("en")
    create_bar_chart("ru")
    create_radar_chart("en")
    create_radar_chart("ru")
    print("Charts generated in docs/assets/")
