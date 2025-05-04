import matplotlib.pyplot as plt
import scienceplots

def set_style():
    plt.style.use(['science', 'grid'])
    plt.rcParams['text.usetex'] = False
    plt.rcParams['image.cmap'] = 'cividis'
    
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 8  
    plt.rcParams['ytick.labelsize'] = 8 
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['axes.titlesize'] = 10
    
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300 
    
    plt.rcParams['axes.grid'] = True  
    plt.rcParams['axes.spines.top'] = True 
    plt.rcParams['axes.spines.right'] = True
    plt.rcParams['xtick.top'] = False  
    plt.rcParams['ytick.right'] = False
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['ytick.left'] = False