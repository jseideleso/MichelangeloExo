import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def set_plot_defaults():
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams['font.size']           = 18
    plt.rcParams['xtick.labelsize']     = 16
    plt.rcParams['ytick.labelsize']     = 16
    plt.rcParams['legend.fontsize']     = 16
    plt.rcParams['axes.titlesize']      = 16
    plt.rcParams['axes.labelsize']      = 16
    plt.rcParams['axes.linewidth']      = 1.5
    plt.rcParams['lines.linewidth']     = 1.5
    plt.rcParams['xtick.bottom']        = 'True'
    plt.rcParams['xtick.direction']     = 'out'
    plt.rcParams['ytick.left']          = 'True'
    plt.rcParams['ytick.direction']     = 'out'
    plt.rcParams['xtick.major.width']   = 1.5
    plt.rcParams['xtick.minor.width']   = 1
    plt.rcParams['xtick.major.size']    = 4
    plt.rcParams['xtick.minor.size']    = 3
    plt.rcParams['ytick.major.width']   = 1.5
    plt.rcParams['ytick.minor.width']   = 1
    plt.rcParams['ytick.major.size']    = 4
    plt.rcParams['ytick.minor.size']    = 3

# ── Load data ────────────────────────────────────────────────────────────────
full  = json.load(open('SNR_comparison_data_L2800-4200nm_R50000_M4500-5500nm_R100000.txt'))
lonly = json.load(open('SNR_comparison_data_Lband_only_2800-4200nm_R50000.txt'))
monly = json.load(open('SNR_comparison_data_Mband_only_4500-5500nm_R100000.txt'))

mols      = ['12C-1H4', '1H2-32S', '28Si-16O', '14N-1H3', '12C-16O', '1H2-16O', '12C-16O2']
mol_names = [r'CH$_4$', r'H$_2$S', r'SiO', r'NH$_3$', r'CO', r'H$_2$O', r'CO$_2$']
cols      = ['teal', 'purple', 'darkorange', 'crimson', 'blue', 'cyan', 'magenta']
temps     = [600, 1100, 1700, 2200]

# x-positions: 4 groups of 4, with a half-unit gap between groups
gap = 0.1
plot_trans = np.array([0, 1, 2, 3])
plot_sp_em = plot_trans + 4 + gap
plot_lonly = plot_sp_em + 4 + gap
plot_monly = plot_lonly + 4 + gap

set_plot_defaults()
fig, ax = plt.subplots(figsize=(14, 8))

def masked(arr):
    """Return array with values < 1 replaced by NaN so lines break at those points."""
    a = np.array(arr, dtype=float)
    a[a < 1.0] = np.nan
    return a

for j, mol in enumerate(mols):
    kw = dict(color=cols[j], marker='o')

    snr_trans = masked(np.array(full['modes']['trans']['snr_improvement'])[:, j])
    ax.errorbar(plot_trans, snr_trans, label=mol_names[j], **kw)

    snr_sp = masked(np.array(full['modes']['sp_em']['snr_improvement'])[:, j])
    ax.errorbar(plot_sp_em, snr_sp, **kw)

    snr_l = masked(np.array(lonly['modes']['trans']['snr_improvement'])[:, j])
    ax.errorbar(plot_lonly, snr_l, **kw)

    snr_m = masked(np.array(monly['modes']['trans']['snr_improvement'])[:, j])
    ax.errorbar(plot_monly, snr_m, **kw)

# Background shading
ylo, yhi = -0.1, 100
ax.fill_between([plot_trans[0]-0.4, plot_trans[-1]+0.4],  ylo, yhi, color='darkorange', alpha=0.2)
ax.fill_between([plot_sp_em[0]-0.4, plot_sp_em[-1]+0.4],  ylo, yhi, color='crimson',    alpha=0.2)
ax.fill_between([plot_lonly[0]-0.4, plot_lonly[-1]+0.4],  ylo, yhi, color='steelblue',  alpha=0.2)
ax.fill_between([plot_monly[0]-0.4, plot_monly[-1]+0.4],  ylo, yhi, color='seagreen',   alpha=0.2)

# Panel labels at the top of each region
ymax = 7
label_y = ymax * 0.94
ax.text(plot_trans.mean(), label_y, 'Transmission',         ha='center', va='top', fontsize=17, fontweight='bold')
ax.text(plot_sp_em.mean(), label_y, 'Emission',             ha='center', va='top', fontsize=17, fontweight='bold')
ax.text(plot_lonly.mean(), label_y, 'L-band only\n(trans.)',ha='center', va='top', fontsize=17, fontweight='bold')
ax.text(plot_monly.mean(), label_y, 'M-band only\n(trans.)',ha='center', va='top', fontsize=17, fontweight='bold')

# Axes
ax.set_ylim(0, ymax)
xmin = plot_trans[0] - 0.5
xmax = plot_monly[-1] + 0.5
ax.set_xlim(xmin, xmax)
ax.set_ylabel('S/N improvement on CRIRES+')
ax.set_xlabel('T [K]')

# X-ticks: temperature numbers only
all_x      = np.concatenate([plot_trans, plot_sp_em, plot_lonly, plot_monly])
all_labels = [str(t) for t in temps] * 4
ax.set_xticks(all_x)
ax.set_xticklabels(all_labels)

# Legend: below the plot, centred
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22),
          ncol=7, frameon=True, framealpha=0.9,
          handlelength=1.5, columnspacing=1.0)

plt.tight_layout()
OUT = 'SNR_coverage_comparison_v3.png'
plt.savefig(OUT, dpi=300, bbox_inches='tight')
print(f'Saved: {OUT}')
