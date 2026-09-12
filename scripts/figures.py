import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, NullFormatter

os.makedirs('figures', exist_ok=True)

models = ['PTO\n(MSE)', 'SSPO\n(surrogate)', 'MSA-NeOpt\n(SPO+, CNN)', 'GRU-NeOpt\n(SPO+, RNN)', 'NeOpt\n(SPO+, Attn)', 'SARIMA‡\n(classical)']
colors = ['gray', 'lightgray', 'green', 'tab:blue', 'orange', 'brown']

regret = np.array([1.35, 0.97, 0.20, 0.25, 0.27, 4.14])
std    = np.array([0.11, 0.08, 0.22, 0.12, 0.30, 6.26])

floor = 0.01
lower_err = np.minimum(std, regret - floor)
yerr = np.vstack([lower_err, std])

fig, ax = plt.subplots(figsize=(9, 6))
ax.bar(models, regret, yerr=yerr, capsize=5, color=colors)

ax.set_yscale('log')
ax.set_ylim(floor, 12)
ax.axhspan(max(floor, 0.20 - 0.22), 0.27 + 0.30, color='green', alpha=0.08)

for i, v in enumerate(regret):
    ax.text(i, (v + std[i]) * 1.15, f'{v:.2f}%', ha='center', fontweight='bold', fontsize=9)

# kill minor ticks entirely, hardcode clean labels on major ticks only
ax.yaxis.set_minor_locator(NullLocator())
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_yticks([0.2, 0.5, 1, 2, 4, 10])
ax.set_yticklabels(['0.2', '0.5', '1', '2', '4', '10'])

ax.set_ylabel('Mean Relative Regret (%) [3-seed mean ± std]')
ax.set_title('Regret Across All SPO+ Variations vs. Non-SPO+ Baselines and SARIMA\n(EirGrid test set)')
plt.tight_layout()
plt.savefig('figures/fig_regret_with_sarima.png', dpi=200)
plt.close()
print("Saved.")