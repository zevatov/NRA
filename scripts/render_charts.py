import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import os
import matplotlib as mpl

# Premium Dark Mode Theme
BG_COLOR = "#0D1117"
PANEL_COLOR = "#161B22"
TEXT_COLOR = "#E6EDF3"
ACCENT_PURPLE = "#A371F7" # Bright Purple for NRA
DARK_PURPLE = "#6E40C9"   # Darker Purple for border
MUTED_GREY = "#8B949E"    # Grey for legacy formats
GRID_COLOR = "#30363D"

mpl.rcParams['text.color'] = TEXT_COLOR
mpl.rcParams['axes.labelcolor'] = TEXT_COLOR
mpl.rcParams['xtick.color'] = TEXT_COLOR
mpl.rcParams['ytick.color'] = TEXT_COLOR
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['JetBrains Mono', 'Fira Code', 'Inter', 'Roboto', 'Arial']

# Set animation params
FRAMES = 60
FPS = 30

def create_bar_chart(lang="en", animated=True):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    formats = ['NRA', 'ZIP', 'Tar.gz']
    packing_times = np.array([3.3, 13.4, 38.0])
    y_pos = np.arange(len(formats))
    
    # Initialize empty bars if animated, full bars if static
    initial_times = [0]*len(formats) if animated else packing_times
    glow_bars = ax.barh(y_pos, initial_times, alpha=0.3, height=0.6, linewidth=0)
    main_bars = ax.barh(y_pos, initial_times, height=0.4, linewidth=1.5)
    
    # Setup styling
    for i in range(len(formats)):
        color = ACCENT_PURPLE if i == 0 else GRID_COLOR
        edge = DARK_PURPLE if i == 0 else 'none'
        
        glow_bars[i].set_color(color)
        main_bars[i].set_color(color)
        main_bars[i].set_edgecolor(edge)
        
    ax.set_yticks(y_pos)
    labels = ax.set_yticklabels([f"{f}" for i, f in enumerate(formats)], fontweight='bold', fontsize=12)
    for i, label in enumerate(labels):
        label.set_color(ACCENT_PURPLE if formats[i] == 'NRA' else MUTED_GREY)
        
    ax.invert_yaxis()
    ax.set_xlim(0, 40)
    
    title = 'Packing Time (60,000 files)' if lang == "en" else 'Время упаковки (60,000 файлов)'
    ax.set_title(title, pad=20, fontsize=16, fontweight='bold', color=TEXT_COLOR)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.grid(axis='x', color=GRID_COLOR, linestyle='--', alpha=0.7)
    
    # Texts
    texts = [ax.text(packing_times[i] + 0.5 if not animated else 0, y_pos[i], f"{packing_times[i]}s" if not animated else "", va='center', color=ACCENT_PURPLE if i==0 else MUTED_GREY, fontweight='bold') for i in range(len(formats))]

    suffix = "" if lang == "en" else "_ru"
    
    if animated:
        def update(frame):
            progress = 1 - (1 - frame/FRAMES)**3
            current_times = packing_times * progress
            
            for i in range(len(formats)):
                glow_bars[i].set_width(current_times[i])
                main_bars[i].set_width(current_times[i])
                if current_times[i] > 0.5:
                    texts[i].set_position((current_times[i] + 0.5, y_pos[i]))
                    texts[i].set_text(f"{current_times[i]:.1f}s")
                    
            return list(glow_bars) + list(main_bars) + texts

        out_path = f"../docs/assets/archiver_benchmark{suffix}.gif"
        anim = FuncAnimation(fig, update, frames=FRAMES + 15, interval=1000/FPS, blit=False)
        anim.save(out_path, writer=PillowWriter(fps=FPS))
    else:
        out_path = f"../docs/assets/archiver_benchmark{suffix}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, facecolor=BG_COLOR, bbox_inches='tight', transparent=False)
        
    plt.close()


def create_radar_chart(lang="en", animated=True):
    categories = ['Cloud Streaming', 'Random Access', 'PyTorch Native', 'Deduplication', 'Encryption', 'Fault Tolerance']
    if lang == "ru":
        categories = ['Cloud Streaming', 'Случайный доступ', 'PyTorch Native', 'Дедупликация', 'Шифрование', 'Отказоустойчивость']
        
    N = len(categories)
    
    nra_vals = np.array([10, 10, 10, 9, 10, 9])
    tar_vals = np.array([0, 1, 3, 0, 0, 5])
    webdataset = np.array([8, 2, 9, 0, 0, 6])
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    
    nra_vals = np.append(nra_vals, nra_vals[0])
    tar_vals = np.append(tar_vals, tar_vals[0])
    webdataset = np.append(webdataset, webdataset[0])
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), facecolor=BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    ini_nra = nra_vals if not animated else np.zeros_like(nra_vals)
    ini_tar = tar_vals if not animated else np.zeros_like(tar_vals)
    ini_wd = webdataset if not animated else np.zeros_like(webdataset)
    
    line_nra, = ax.plot(angles if not animated else [], ini_nra if not animated else [], linewidth=2, linestyle='solid', color=ACCENT_PURPLE, label='NRA v4.5')
    if animated:
        fill_nra = ax.fill([], [], color=ACCENT_PURPLE, alpha=0.4)[0]
    else:
        fill_nra = ax.fill(angles, ini_nra, color=ACCENT_PURPLE, alpha=0.4)[0]
    
    line_wd, = ax.plot(angles if not animated else [], ini_wd if not animated else [], linewidth=1.5, linestyle='dashed', color=MUTED_GREY, label='WebDataset')
    line_tar, = ax.plot(angles if not animated else [], ini_tar if not animated else [], linewidth=1.5, linestyle='dotted', color=GRID_COLOR, label='Tar.gz')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color=TEXT_COLOR, size=11)
    
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels([])
    ax.set_ylim(0, 10)
    
    ax.spines['polar'].set_color(GRID_COLOR)
    ax.grid(color=GRID_COLOR, linestyle='--')
    
    title = 'NRA vs Legacy Formats' if lang == "en" else 'NRA против устаревших форматов'
    plt.title(title, size=16, color=TEXT_COLOR, y=1.1, fontweight='bold')
    
    legend = ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), facecolor=PANEL_COLOR, edgecolor=GRID_COLOR)
    texts = legend.get_texts()
    texts[0].set_color(ACCENT_PURPLE) # NRA
    texts[1].set_color(MUTED_GREY) # WebDataset
    texts[2].set_color(GRID_COLOR) # Tar.gz

    suffix = "" if lang == "en" else "_ru"

    if animated:
        def update(frame):
            progress = 1 - (1 - frame/FRAMES)**3
            
            c_nra = nra_vals * progress
            c_wd = webdataset * progress
            c_tar = tar_vals * progress
            
            line_nra.set_data(angles, c_nra)
            fill_nra.set_xy(np.column_stack((angles, c_nra)))
            
            line_wd.set_data(angles, c_wd)
            line_tar.set_data(angles, c_tar)
            
            return line_nra, fill_nra, line_wd, line_tar

        out_path = f"../docs/assets/radar{suffix}.gif"
        anim = FuncAnimation(fig, update, frames=FRAMES + 15, interval=1000/FPS, blit=False)
        anim.save(out_path, writer=PillowWriter(fps=FPS))
    else:
        out_path = f"../docs/assets/radar{suffix}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, facecolor=BG_COLOR, bbox_inches='tight', transparent=False)
        
    plt.close()

if __name__ == "__main__":
    os.makedirs("../docs/assets", exist_ok=True)
    # Generate Animated GIFs for README
    create_bar_chart("en", animated=True)
    create_bar_chart("ru", animated=True)
    create_radar_chart("en", animated=True)
    create_radar_chart("ru", animated=True)
    
    # Generate Static PNGs for Whitepaper
    create_bar_chart("en", animated=False)
    create_bar_chart("ru", animated=False)
    create_radar_chart("en", animated=False)
    create_radar_chart("ru", animated=False)
    print("Animated GIFs and Static PNGs generated in docs/assets/")
