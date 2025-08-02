import matplotlib.pyplot as plt
import scienceplots

def set_style(axes_labelsize=8, tick_labelsize=8, legend_fontsize=8, axes_titlesize=10, 
              linewidth=1.5, savefig_dpi=300, figure_dpi=300, 
              grid=True, spines_top=True, spines_right=True,
              xtick_top=False, ytick_right=False, spines_left=True, ytick_left=False):
    
    plt.style.use(['science', 'grid'])
    plt.rcParams['text.usetex'] = True
    plt.rcParams['image.cmap'] = 'cividis'
    
    plt.rcParams['axes.labelsize'] = axes_labelsize
    plt.rcParams['xtick.labelsize'] = tick_labelsize
    plt.rcParams['ytick.labelsize'] = tick_labelsize
    plt.rcParams['legend.fontsize'] = legend_fontsize
    plt.rcParams['axes.titlesize'] = axes_titlesize
    
    plt.rcParams['lines.linewidth'] = linewidth
    plt.rcParams['savefig.dpi'] = savefig_dpi
    plt.rcParams['figure.dpi'] = figure_dpi
    
    plt.rcParams['axes.grid'] = grid
    plt.rcParams['axes.spines.top'] = spines_top
    plt.rcParams['axes.spines.right'] = spines_right
    plt.rcParams['xtick.top'] = xtick_top
    plt.rcParams['ytick.right'] = ytick_right
    plt.rcParams['axes.spines.left'] = spines_left
    plt.rcParams['ytick.left'] = ytick_left