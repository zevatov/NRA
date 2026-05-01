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
FPS = 30
PAUSE_FRAMES = 150 # 5 seconds pause at the end

def ease_out_cubic(x):
    return 1 - (1 - x)**3

def create_radar_chart(lang="en", animated=True):
    categories = ['Cloud Streaming', 'Random Access', 'Storage Efficiency', 'Simplicity', 'Data Universality', 
                  'Fault Tolerance', 'Encryption (AES)', 'Delta Updates', 'PyTorch Integration']
    if lang == "ru":
        categories = ['Cloud Streaming', 'Random Access', 'Storage Efficiency', 'Simplicity', 'Data Universality', 
                      'Fault Tolerance', 'Encryption (AES)', 'Delta Updates', 'PyTorch Integration']
                      
    N = len(categories)
    
    # Values from 1 to 5
    data = {
        'NRA v4.5':           np.array([4, 5, 5, 3, 5, 4, 5, 5, 5]),
        'WebDataset':         np.array([3, 1, 1, 3, 4, 3, 1, 1, 5]),
        'TFRecord / Parquet': np.array([1, 2, 3, 2, 2, 3, 2, 3, 4]),
        'Tar.gz':             np.array([1, 1, 4, 4, 5, 1, 1, 1, 2]),
        'Classic Tar':        np.array([1, 1, 1, 4, 5, 1, 1, 1, 2]),
        'Raw Disk / S3':      np.array([1, 5, 1, 5, 5, 3, 1, 5, 3])
    }
    
    colors = {
        'NRA v4.5': ACCENT_PURPLE,
        'WebDataset': '#D29922', # Orange/Yellowish
        'TFRecord / Parquet': '#238636', # Green
        'Tar.gz': '#58A6FF', # Blue
        'Classic Tar': '#F85149', # Red
        'Raw Disk / S3': MUTED_GREY
    }
    
    styles = {
        'NRA v4.5': 'solid',
        'WebDataset': 'dashed',
        'TFRecord / Parquet': 'dashdot',
        'Tar.gz': 'dotted',
        'Classic Tar': 'dotted',
        'Raw Disk / S3': 'solid'
    }
    
    linewidths = {
        'NRA v4.5': 3.0,
        'WebDataset': 1.5,
        'TFRecord / Parquet': 1.5,
        'Tar.gz': 1.5,
        'Classic Tar': 1.5,
        'Raw Disk / S3': 1.5
    }

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1] # close the loop
    
    for key in data:
        data[key] = np.append(data[key], data[key][0])
        
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True), facecolor=BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    # Add padding to labels so they don't overlap
    ax.tick_params(pad=30)
    
    lines = {}
    fills = {}
    
    for name in data:
        lines[name], = ax.plot([], [], linewidth=linewidths[name], linestyle=styles[name], color=colors[name], label=name)
        if name == 'NRA v4.5':
            fills[name] = ax.fill([], [], color=colors[name], alpha=0.3)[0]

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color=TEXT_COLOR, size=12)
    
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], color=GRID_COLOR)
    ax.set_ylim(0, 5)
    
    ax.spines['polar'].set_color(GRID_COLOR)
    ax.grid(color=GRID_COLOR, linestyle='--', alpha=0.5, zorder=10)
    ax.set_axisbelow(False) # Draw grid lines on top of patches
    
    title = 'NRA vs Legacy Formats' if lang == "en" else 'NRA против устаревших форматов'
    plt.title(title, size=20, color=TEXT_COLOR, y=1.1, fontweight='bold')
    
    legend = ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, fontsize=11)
    for text in legend.get_texts():
        if text.get_text() == 'NRA v4.5':
            text.set_color(ACCENT_PURPLE)
            text.set_fontweight('bold')
        else:
            text.set_color(MUTED_GREY)

    suffix = "" if lang == "en" else "_ru"

    if animated:
        # Sequence:
        # NRA (0-30), WebDataset (30-60), Parquet (60-90), Tar.gz (90-120), Tar (120-150), Raw (150-180)
        frames_per_format = 30
        format_keys = list(data.keys())
        total_anim_frames = frames_per_format * len(format_keys)
        
        def update(frame):
            for i, name in enumerate(format_keys):
                start_f = i * frames_per_format
                end_f = start_f + frames_per_format
                
                if frame < start_f:
                    prog = 0
                elif frame > end_f:
                    prog = 1
                else:
                    prog = ease_out_cubic((frame - start_f) / frames_per_format)
                
                if prog > 0:
                    c_vals = data[name] * prog
                    lines[name].set_data(angles, c_vals)
                    if name == 'NRA v4.5':
                        fills[name].set_xy(np.column_stack((angles, c_vals)))
            
            return list(lines.values()) + list(fills.values())

        out_path = f"../docs/assets/radar{suffix}.gif"
        anim = FuncAnimation(fig, update, frames=total_anim_frames + PAUSE_FRAMES, interval=1000/FPS, blit=False)
        anim.save(out_path, writer=PillowWriter(fps=FPS))
    else:
        for name in data:
            lines[name].set_data(angles, data[name])
            if name == 'NRA v4.5':
                fills[name].set_xy(np.column_stack((angles, data[name])))
                
        out_path = f"../docs/assets/radar{suffix}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, facecolor=BG_COLOR, bbox_inches='tight', transparent=False)
        
    plt.close()

def create_bar_chart(lang="en", animated=True):
    # Two subplots: Time and Size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    
    formats = ['NRA', 'TAR', 'TAR.GZ', 'ZIP', '7Z', 'RAR']
    
    # Approximated data based on typical archiver performance
    pack_time = np.array([3.3, 1.5, 38.0, 13.4, 120.0, 45.0])
    unpack_time = np.array([0.0, 1.5, 8.0, 5.0, 15.0, 10.0]) # 0 for NRA (zero-copy)
    sizes = np.array([140, 450, 150, 160, 110, 130])
    
    y_pos = np.arange(len(formats))
    
    for ax in (ax1, ax2):
        ax.set_facecolor(BG_COLOR)
        ax.set_yticks(y_pos)
        labels = ax.set_yticklabels(formats, fontweight='bold', fontsize=12)
        for i, label in enumerate(labels):
            label.set_color(ACCENT_PURPLE if formats[i] == 'NRA' else MUTED_GREY)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(GRID_COLOR)
        ax.spines['bottom'].set_color(GRID_COLOR)
        ax.grid(axis='x', color=GRID_COLOR, linestyle='--', alpha=0.7)
    
    title1 = 'Time (Seconds)' if lang == "en" else 'Время (Секунды)'
    title2 = 'Archive Size (MB)' if lang == "en" else 'Размер Архива (МБ)'
    
    ax1.set_title(title1, pad=20, fontsize=16, fontweight='bold', color=TEXT_COLOR)
    ax2.set_title(title2, pad=20, fontsize=16, fontweight='bold', color=TEXT_COLOR)
    
    ax1.set_xlim(0, 130)
    ax2.set_xlim(0, 500)
    
    # Initialize bars
    bars_pack = ax1.barh(y_pos, [0]*len(formats), height=0.35, align='center', color=MUTED_GREY, label='Packing')
    bars_unpack = ax1.barh(y_pos + 0.35, [0]*len(formats), height=0.35, align='center', color=GRID_COLOR, label='Unpacking')
    bars_size = ax2.barh(y_pos, [0]*len(formats), height=0.6, align='center', color=MUTED_GREY)
    
    texts_pack = [ax1.text(0, y_pos[i], "", va='center', color=ACCENT_PURPLE if formats[i]=='NRA' else MUTED_GREY, fontweight='bold') for i in range(len(formats))]
    texts_unpack = [ax1.text(0, y_pos[i] + 0.35, "", va='center', color=DARK_PURPLE if formats[i]=='NRA' else MUTED_GREY, fontweight='bold', fontsize=10) for i in range(len(formats))]
    texts_size = [ax2.text(0, y_pos[i], "", va='center', color=ACCENT_PURPLE if formats[i]=='NRA' else MUTED_GREY, fontweight='bold') for i in range(len(formats))]

    for i in range(len(formats)):
        if formats[i] == 'NRA':
            bars_pack[i].set_color(ACCENT_PURPLE)
            bars_pack[i].set_edgecolor(DARK_PURPLE)
            bars_unpack[i].set_color(DARK_PURPLE)
            bars_size[i].set_color(ACCENT_PURPLE)
            bars_size[i].set_edgecolor(DARK_PURPLE)

    ax1.legend(facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    
    suffix = "" if lang == "en" else "_ru"

    if animated:
        frames_per_format = 20
        total_anim_frames = frames_per_format * len(formats)
        
        def update(frame):
            for i in range(len(formats)):
                start_f = i * frames_per_format
                end_f = start_f + frames_per_format
                
                if frame < start_f:
                    prog = 0
                elif frame > end_f:
                    prog = 1
                else:
                    prog = ease_out_cubic((frame - start_f) / frames_per_format)
                
                cur_pack = pack_time[i] * prog
                cur_unpack = unpack_time[i] * prog
                cur_size = sizes[i] * prog
                
                bars_pack[i].set_width(cur_pack)
                bars_unpack[i].set_width(cur_unpack)
                bars_size[i].set_width(cur_size)
                
                if cur_pack > 0.5:
                    texts_pack[i].set_position((cur_pack + 2, y_pos[i]))
                    texts_pack[i].set_text(f"{cur_pack:.1f}s")
                if unpack_time[i] == 0.0 and prog > 0.5:
                    texts_unpack[i].set_position((2, y_pos[i] + 0.35))
                    texts_unpack[i].set_text("0.0s (Zero-Disk)")
                elif cur_unpack > 0.5:
                    texts_unpack[i].set_position((cur_unpack + 2, y_pos[i] + 0.35))
                    texts_unpack[i].set_text(f"{cur_unpack:.1f}s")
                if cur_size > 0.5:
                    texts_size[i].set_position((cur_size + 5, y_pos[i]))
                    texts_size[i].set_text(f"{int(cur_size)}MB")
                
            return list(bars_pack) + list(bars_unpack) + list(bars_size) + texts_pack + texts_unpack + texts_size

        out_path = f"../docs/assets/archiver_benchmark{suffix}.gif"
        anim = FuncAnimation(fig, update, frames=total_anim_frames + PAUSE_FRAMES, interval=1000/FPS, blit=False)
        anim.save(out_path, writer=PillowWriter(fps=FPS))
    else:
        for i in range(len(formats)):
            bars_pack[i].set_width(pack_time[i])
            bars_unpack[i].set_width(unpack_time[i])
            bars_size[i].set_width(sizes[i])
            texts_pack[i].set_position((pack_time[i] + 2, y_pos[i]))
            texts_pack[i].set_text(f"{pack_time[i]:.1f}s")
            if unpack_time[i] == 0.0:
                texts_unpack[i].set_position((2, y_pos[i] + 0.35))
                texts_unpack[i].set_text("0.0s (Zero-Disk)")
            elif unpack_time[i] > 0:
                texts_unpack[i].set_position((unpack_time[i] + 2, y_pos[i] + 0.35))
                texts_unpack[i].set_text(f"{unpack_time[i]:.1f}s")
            texts_size[i].set_position((sizes[i] + 5, y_pos[i]))
            texts_size[i].set_text(f"{int(sizes[i])}MB")
            
        out_path = f"../docs/assets/archiver_benchmark{suffix}.png"
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
