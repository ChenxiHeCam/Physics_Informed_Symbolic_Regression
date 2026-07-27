"""Figure 2 (PhyE2E-style, 4 panels) for _PAPER_v3_phye2e.md — FINAL locked numbers.
2a manifold separability (AUC by control) | 2b accuracy bars | 2c margin over 2nd-best | 2d speed-accuracy.
All values are the final audited results. Output: _benchmark_results/figs/fig2_results.png"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': 0.9, 'figure.dpi': 200})

OURS = '#0F766E'      # teal — PISR accent, consistent across all panels
BASE = ['#64748B', '#94A3B8', '#CBD5E1', '#E2E8F0']  # greys for baselines
METHODS = ['PISR', 'PySR', 'PSE', 'GenSR', 'PhyE2E']
COLORS = [OURS] + BASE

# ---- data (final locked) ----
acc = {'real-physics\n(797)': [51.3, 43.7, 33.1, 16.3, 14.4],
       'cross-domain\n(270)': [71.1, 55.2, 50.7, 35.6, 33.7],
       'Feynman\n(119)':      [39.5, 31.9, 32.8, 10.1, 2.5]}
margins = {'real-physics': 7.6, 'cross-domain': 15.9, 'Feynman': 6.7}  # ours - 2nd best
speed = {'PISR': (12.25, 51.3), 'PySR': (30, 43.7), 'PSE': (30, 33.1),
         'GenSR': (90, 16.3), 'PhyE2E': (3.3, 14.4)}
auc_ctrl = ['unconstrained', 'matched\nop-freq + length', 'RHS-only\n(matched, strict)']
auc_val = [0.996, 0.997, 0.944]

fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.7))

# --- 2a: manifold separability (AUC by control) ---
a = ax[0]
bars = a.bar(range(3), auc_val, color=['#94A3B8', '#94A3B8', OURS], width=0.62, zorder=3)
a.axhline(0.5, ls='--', lw=1, color='#9CA3AF', zorder=2)
a.text(2.35, 0.515, 'chance', fontsize=8, color='#6B7280', ha='right')
for i, v in enumerate(auc_val):
    a.text(i, v + 0.006, f'{v:.3f}', ha='center', fontsize=9.5, fontweight='bold')
a.set_xticks(range(3)); a.set_xticklabels(auc_ctrl, fontsize=8.2)
a.set_ylim(0.45, 1.02); a.set_ylabel('linear-probe ROC-AUC')
a.set_title('Real laws vs random expressions', fontsize=10.5, pad=8)
a.text(0.02, 0.06, 'MMD$^2$ = 0.109\nparticip. ratio 21.5 / 10.5',
       transform=a.transAxes, fontsize=8.2, color='#374151',
       bbox=dict(boxstyle='round,pad=0.35', fc='#F1F5F9', ec='none'))

# --- 2b: accuracy bars, 3 benchmarks x 5 methods ---
b = ax[1]
groups = list(acc.keys()); ng = len(groups); nm = len(METHODS)
x = np.arange(ng); w = 0.16
for j in range(nm):
    vals = [acc[g][j] for g in groups]
    b.bar(x + (j - (nm - 1) / 2) * w, vals, w, color=COLORS[j], label=METHODS[j], zorder=3,
          edgecolor='white', linewidth=0.4)
b.set_xticks(x); b.set_xticklabels(groups, fontsize=9)
b.set_ylabel('structural accuracy (%)'); b.set_ylim(0, 80)
b.set_title('Accuracy across benchmarks', fontsize=10.5, pad=8)
b.legend(fontsize=7.8, frameon=False, ncol=1, loc='upper right', handlelength=1.1)
for j in range(nm):  # label PISR bars
    b.text(x[0] + (j - (nm - 1) / 2) * w if False else 0, 0, '')
for gi, g in enumerate(groups):
    v = acc[g][0]
    b.text(gi - (nm - 1) / 2 * w, v + 1, f'{v:.1f}', ha='center', fontsize=8, fontweight='bold', color=OURS)

# --- 2c: margin over second-best ---
c = ax[2]
mk = list(margins.keys()); mv = [margins[k] for k in mk]
barsc = c.bar(range(3), mv, color=[OURS, '#0EA5A0', OURS], width=0.6, zorder=3)
c.bar(1, mv[1], width=0.6, color='#0D9488', zorder=3)  # highlight cross-domain
for i, v in enumerate(mv):
    c.text(i, v + 0.3, f'+{v:.1f}', ha='center', fontsize=10, fontweight='bold')
c.set_xticks(range(3)); c.set_xticklabels(['real-\nphysics', 'cross-\ndomain', 'Feynman'], fontsize=9)
c.set_ylabel('PISR − 2nd best (pp)'); c.set_ylim(0, 19)
c.set_title('Margin over strongest baseline', fontsize=10.5, pad=8)
c.text(1, 16.8, 'largest margin\nis out-of-distribution', ha='center', fontsize=8, color='#0F766E', style='italic')

# --- 2d: speed vs accuracy ---
d = ax[3]
for k, (t, aa) in speed.items():
    col = OURS if k == 'PISR' else '#64748B'
    d.scatter(t, aa, s=130 if k == 'PISR' else 80, color=col, zorder=3,
              edgecolor='white', linewidth=1.2)
    dx = 1.15 if k != 'PhyE2E' else 1.15
    d.annotate(k, (t, aa), (t * dx, aa + 1.3), fontsize=8.5,
               fontweight='bold' if k == 'PISR' else 'normal',
               color=OURS if k == 'PISR' else '#374151')
d.set_xscale('log'); d.set_xlabel('per-task wall-clock (s, log)')
d.set_ylabel('real-physics accuracy (%)'); d.set_ylim(5, 60)
d.set_xlim(2, 160)
d.set_title('Speed–accuracy trade-off', fontsize=10.5, pad=8)
d.text(0.97, 0.06, 'up-and-left = better', transform=d.transAxes, ha='right',
       fontsize=8, color='#6B7280', style='italic')

# panel letters
for k, axx in enumerate(ax):
    axx.text(-0.16, 1.06, chr(97 + k), transform=axx.transAxes, fontsize=14, fontweight='bold', va='top')

plt.tight_layout(w_pad=2.2)
out = 'figures/fig2_results.png'
plt.savefig(out, bbox_inches='tight', facecolor='white')
print('saved', out)
