# generate all plots for the presentation
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import optimize, stats
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

os.makedirs('plots', exist_ok=True)

# --- Load data ---
sig = pd.read_csv('data/signal_Bs2MuMu.txt', sep=r'\s+')
bkg = pd.read_csv('data/background_combinatorial.txt', sep=r'\s+')
# old data had background PT in MeV, new data is already in GeV

features = sig.columns.tolist()
print(f"Loaded {len(sig)} signal, {len(bkg)} background events")

# --- Plot 1: Feature histograms ---
units = {
    'Mu1_PT': '[GeV]', 'Mu2_PT': '[GeV]',
    'Mu1_P': '[GeV]', 'Mu2_P': '[GeV]',
    'tot_PT': '[GeV]', 'VTXCHI2': '',
    'ISO': '', 'MASS': '[GeV]',
}
log_x_features = {'Mu1_PT', 'Mu2_PT', 'Mu1_P', 'Mu2_P', 'tot_PT'}

fig, axes = plt.subplots(2, 4, figsize=(7, 3))
axes = axes.flatten()
for i, feat in enumerate(features):
    ax = axes[i]
    use_log_x = feat in log_x_features
    if use_log_x:
        lo = max(min(sig[feat].min(), bkg[feat].min()), 1e-3)
        hi = max(sig[feat].quantile(0.99), bkg[feat].quantile(0.99))
        bins = np.logspace(np.log10(lo), np.log10(hi), 50)
    else:
        lo = min(sig[feat].min(), bkg[feat].min())
        hi = max(sig[feat].quantile(0.99), bkg[feat].quantile(0.99))
        bins = np.linspace(lo, hi, 50)
    ax.hist(sig[feat], bins=bins, density=True, histtype='stepfilled', alpha=0.5, label='Signal', color='b')
    ax.hist(bkg[feat], bins=bins, density=True, histtype='stepfilled', alpha=0.5, label='Background', color='r')
    if use_log_x:
        ax.set_xscale('log')
    unit = units.get(feat, '')
    ax.set_xlabel(f'{feat} {unit}'.strip())
    ax.set_ylabel('a.u.')
    ax.legend(fontsize=5)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=6)
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(7)
plt.suptitle('Signal vs Background feature distributions', fontsize=10)
plt.tight_layout()
plt.savefig('plots/feature_histograms.png', dpi=200, bbox_inches='tight')
plt.close()
print("  [1/10] feature_histograms.png")

# --- Plot 2: Fisher scores ---
fisher_scores = {}
for feat in features:
    mu_s, mu_b = sig[feat].mean(), bkg[feat].mean()
    var_s, var_b = sig[feat].var(), bkg[feat].var()
    fisher_scores[feat] = (mu_s - mu_b)**2 / (var_s + var_b)

fisher_df = pd.DataFrame({
    'Feature': fisher_scores.keys(),
    'Fisher Score': fisher_scores.values()
}).sort_values('Fisher Score', ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(4, 3))
colors = ['b' if f != 'MASS' else 'r' for f in fisher_df['Feature']]
ax.barh(fisher_df['Feature'], fisher_df['Fisher Score'], color=colors)
ax.set_xlabel('Fisher Score')
ax.set_title('Fisher Score Ranking')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')
ax.text(0.95, 0.05, 'Red = excluded from BDT', transform=ax.transAxes,
        ha='right', fontsize=7, color='r')
plt.tight_layout()
plt.savefig('plots/fisher_scores.png', dpi=200, bbox_inches='tight')
plt.close()
print("  [2/10] fisher_scores.png")

bdt_features_3 = [f for f in fisher_df['Feature'] if f != 'MASS'][:3]

# --- Train BDTs ---
X_all = pd.concat([sig, bkg], ignore_index=True)
y_all = np.concatenate([np.ones(len(sig)), np.zeros(len(bkg))])

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.5, random_state=42
)

bdt_3 = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
bdt_3.fit(X_train[bdt_features_3], y_train)

bdt_features_7 = [f for f in features if f != 'MASS']
bdt_7 = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
bdt_7.fit(X_train[bdt_features_7], y_train)

acc_3 = accuracy_score(y_test, bdt_3.predict(X_test[bdt_features_3]))
acc_7 = accuracy_score(y_test, bdt_7.predict(X_test[bdt_features_7]))
print(f"  BDT accuracies: 3-feat={acc_3:.4f}, 7-feat={acc_7:.4f}")

# --- Plot 3: BDT feature importances ---
importances = bdt_7.feature_importances_
sorted_idx = np.argsort(importances)

fig, ax = plt.subplots(figsize=(4, 3))
ax.barh(np.array(bdt_features_7)[sorted_idx], importances[sorted_idx], color='b')
ax.set_xlabel('Feature Importance')
ax.set_title('BDT Feature Importances (7 features)')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('plots/bdt_importances.png', dpi=200, bbox_inches='tight')
plt.close()
print("  [3/10] bdt_importances.png")

# --- Plot 4: BDT score distribution ---
scores_sig = bdt_7.decision_function(X_test[bdt_features_7][y_test == 1])
scores_bkg = bdt_7.decision_function(X_test[bdt_features_7][y_test == 0])

fig, ax = plt.subplots(figsize=(4, 3))
lo_s = min(scores_sig.min(), scores_bkg.min())
hi_s = max(scores_sig.max(), scores_bkg.max())
bins_score = np.linspace(lo_s, hi_s, 50)
ax.hist(scores_sig, bins=bins_score, density=True, histtype='stepfilled',
        alpha=0.5, label='Signal (test)', color='b')
ax.hist(scores_bkg, bins=bins_score, density=True, histtype='stepfilled',
        alpha=0.5, label='Background (test)', color='r')
ax.set_xlabel('BDT Output Score')
ax.set_ylabel('a.u.')
ax.set_title('BDT classifier output (7 features)')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/bdt_score_dist.png', dpi=200, bbox_inches='tight')
plt.close()
print("  [4/10] bdt_score_dist.png")

# --- Plot 5: Mass distribution after BDT ---
# evaluate on test set only to avoid overfitting bias
sig_test = X_test[y_test == 1]
bkg_test = X_test[y_test == 0]

sig_pred = bdt_7.predict(sig_test[bdt_features_7])
bkg_pred = bdt_7.predict(bkg_test[bdt_features_7])
eps_s = np.mean(sig_pred == 1)
eps_b = np.mean(bkg_pred == 1)

sig_mass_after = sig_test.loc[sig_pred == 1, 'MASS'].values
bkg_mass_after = bkg_test.loc[bkg_pred == 1, 'MASS'].values

fig, ax = plt.subplots(figsize=(4, 3))
bins = np.linspace(4.0, 6.0, 50)
ax.hist(sig_mass_after, bins=bins, histtype='stepfilled',
        alpha=0.5, label=f'Signal ({len(sig_mass_after)} events)', color='b')
ax.hist(bkg_mass_after, bins=bins, histtype='stepfilled',
        alpha=0.5, label=f'Background ({len(bkg_mass_after)} events)', color='r')
ax.set_xlabel('MASS [GeV]')
ax.set_ylabel('Events / 0.04 GeV')
ax.set_title('Mass distribution after BDT selection (test set)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.text(0.97, 0.95, f'$\\varepsilon_s$ = {eps_s:.4f}\n$\\varepsilon_b$ = {eps_b:.4f}',
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig('plots/mass_after_bdt.png', dpi=200, bbox_inches='tight')
plt.close()
print("  [5/10] mass_after_bdt.png")

# --- Plot 7: Exponential fit to background ---
mass_lo, mass_hi = 4.0, 6.0

def exp_nll(lam, data, lo=4.0, hi=6.0):
    norm_const = (np.exp(-lam * lo) - np.exp(-lam * hi)) / lam
    return -np.sum(-lam * data) + len(data) * np.log(norm_const)

# fit lambda from full background sample as instructed by the assignment
bkg_mass_all = bkg['MASS'].values
result = optimize.minimize_scalar(lambda l: exp_nll(l, bkg_mass_all), bounds=(-5, 5), method='bounded')
lam_fit = result.x

fig, ax = plt.subplots(figsize=(4, 3))
bins = np.linspace(mass_lo, mass_hi, 40)
ax.hist(bkg_mass_all, bins=bins, density=True, histtype='stepfilled',
        alpha=0.5, label='Background MC (full sample)', color='r')
m_plot = np.linspace(mass_lo, mass_hi, 200)
norm_const = (np.exp(-lam_fit * mass_lo) - np.exp(-lam_fit * mass_hi)) / lam_fit
ax.plot(m_plot, np.exp(-lam_fit * m_plot) / norm_const, 'k-', lw=2, label=f'Exp fit (lam={lam_fit:.3f})')
ax.set_xlabel('MASS [GeV]')
ax.set_ylabel('Probability density [1/GeV]')
ax.set_title('Exponential fit to background mass shape')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/exp_fit.png', dpi=200, bbox_inches='tight')
plt.close()
print("  [6/10] exp_fit.png")

# --- Toy generation helpers ---
n_sig_per_year = 50 * eps_s
n_bkg_per_year = 2000 * eps_b

def sample_truncated_exp(lam, lo, hi, size):
    # inverse CDF sampling
    u = np.random.uniform(0, 1, size)
    cdf_lo = np.exp(-lam * lo)
    cdf_hi = np.exp(-lam * hi)
    return -np.log(cdf_lo - u * (cdf_lo - cdf_hi)) / lam

def sample_truncated_gauss(mu, sigma, lo, hi, size):
    samples = []
    while len(samples) < size:
        s = np.random.normal(mu, sigma, size * 2)
        s = s[(s >= lo) & (s <= hi)]
        samples.extend(s)
    return np.array(samples[:size])

def generate_toy(n_sig, n_bkg, lam, mu=5.0, sigma=0.03, lo=4.0, hi=6.0):
    ns = np.random.poisson(n_sig)
    nb = np.random.poisson(n_bkg)
    sig_events = sample_truncated_gauss(mu, sigma, lo, hi, ns)
    bkg_events = sample_truncated_exp(lam, lo, hi, nb)
    data = np.concatenate([sig_events, bkg_events])
    np.random.shuffle(data)
    return data, ns, nb

# --- Plot 8: Example toy dataset ---
np.random.seed(42)
toy_data, ns_toy, nb_toy = generate_toy(n_sig_per_year, n_bkg_per_year, lam_fit)

fig, ax = plt.subplots(figsize=(4, 3))
bins = np.linspace(mass_lo, mass_hi, 50)
ax.hist(toy_data, bins=bins, histtype='stepfilled', alpha=0.7, color='b',
        label=f'Toy ({len(toy_data)} events)')
ax.set_xlabel('MASS [GeV]')
ax.set_ylabel('Events / 0.04 GeV')
ax.set_title(f'Example toy (1 year): {ns_toy} sig + {nb_toy} bkg')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/toy_example.png', dpi=200, bbox_inches='tight')
plt.close()
print("  [7/10] toy_example.png")

# --- Fit toys ---
def gauss_pdf(m, mu=5.0, sigma=0.03, lo=4.0, hi=6.0):
    norm_const = norm.cdf(hi, mu, sigma) - norm.cdf(lo, mu, sigma)
    return norm.pdf(m, mu, sigma) / norm_const

def exp_pdf(m, lam, lo=4.0, hi=6.0):
    norm_const = (np.exp(-lam * lo) - np.exp(-lam * hi)) / lam
    return np.exp(-lam * m) / norm_const

def composite_nll(params, data):
    f_sig, lam = params
    pdf_vals = f_sig * gauss_pdf(data) + (1 - f_sig) * exp_pdf(data, lam)
    pdf_vals = np.maximum(pdf_vals, 1e-300)
    return -np.sum(np.log(pdf_vals))

def bkg_only_nll(params, data):
    lam = params[0]
    pdf_vals = exp_pdf(data, lam)
    pdf_vals = np.maximum(pdf_vals, 1e-300)
    return -np.sum(np.log(pdf_vals))

def fit_toy(data):
    res_sb = optimize.minimize(composite_nll, x0=[0.1, lam_fit], args=(data,),
                               bounds=[(0, 1), (-10, 10)], method='L-BFGS-B')
    f_fit, lam_fit_toy = res_sb.x
    nll_sb = res_sb.fun
    res_b = optimize.minimize(bkg_only_nll, x0=[lam_fit], args=(data,),
                              bounds=[(-10, 10)], method='L-BFGS-B')
    nll_b = res_b.fun
    return f_fit, lam_fit_toy, nll_sb, nll_b

print("  Running 1000 toy fits (1 year)...")
np.random.seed(123)
n_toys = 1000
f_sig_fits = []
significances = []

for i in range(n_toys):
    toy, _, _ = generate_toy(n_sig_per_year, n_bkg_per_year, lam_fit)
    f_fit, lam_toy, nll_sb, nll_b = fit_toy(toy)
    f_sig_fits.append(f_fit)
    q = 2 * (nll_b - nll_sb)
    q = max(q, 0)
    significances.append(np.sqrt(q))

f_sig_fits = np.array(f_sig_fits)
significances = np.array(significances)
f_sig_true = n_sig_per_year / (n_sig_per_year + n_bkg_per_year)

# --- Plot 9: Signal fraction distribution ---
fig, ax = plt.subplots(figsize=(4, 3))
ax.hist(f_sig_fits, bins=40, alpha=0.7, color='b')
ax.axvline(f_sig_true, color='r', ls='--', lw=2, label=f'True f_sig = {f_sig_true:.4f}')
ax.axvline(np.mean(f_sig_fits), color='k', ls='-', lw=2, label=f'Mean fit = {np.mean(f_sig_fits):.4f}')
ax.set_xlabel('Fitted Signal Fraction')
ax.set_ylabel('Pseudo-experiments')
ax.set_title('Fitted signal fraction (1 year, 1000 toys)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/fsig_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print("  [8/10] fsig_distribution.png")

# --- Plot 10: Significance distribution ---
frac_above_5 = np.mean(significances > 5)

fig, ax = plt.subplots(figsize=(4, 3))
counts, bin_edges, _ = ax.hist(significances, bins=40, alpha=0.7, color='b')
ax.axvline(5.0, color='r', ls='--', lw=2, label='5 sigma threshold')
ax.set_xlabel('Significance [sigma]')
ax.set_ylabel('Pseudo-experiments')
ax.set_title(f'Significance distribution (1 year, {n_toys} toys)')
ax.text(0.97, 0.92, f'{frac_above_5*100:.1f}% above 5 sigma',
        transform=ax.transAxes, ha='right', fontsize=9, color='red',
        bbox=dict(facecolor='white', alpha=0.8))
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/significance_dist.png', dpi=200, bbox_inches='tight')
plt.close()
print("  [9/10] significance_dist.png")

# --- Plot 11: Discovery duration scan ---
print("  Scanning durations (0.5-12.5 months, 500 toys each)...")
months_to_scan = np.arange(0.5, 13, 0.5)
n_toys_scan = 500
discovery_fracs = []

np.random.seed(456)
for months in months_to_scan:
    years = months / 12.0
    n_sig_y = n_sig_per_year * years
    n_bkg_y = n_bkg_per_year * years

    n_above_5 = 0
    for _ in range(n_toys_scan):
        toy, _, _ = generate_toy(n_sig_y, n_bkg_y, lam_fit)
        if len(toy) < 2:
            continue
        _, _, nll_sb, nll_b = fit_toy(toy)
        q = max(2 * (nll_b - nll_sb), 0)
        if np.sqrt(q) > 5:
            n_above_5 += 1

    frac = n_above_5 / n_toys_scan
    discovery_fracs.append(frac)
    print(f"    {months:5.1f} months: {frac:.3f}")

discovery_fracs = np.array(discovery_fracs)

idx_95 = np.where(discovery_fracs >= 0.95)[0]
min_months = months_to_scan[idx_95[0]]

fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(months_to_scan, discovery_fracs, 'o-', markersize=4, color='b', lw=1.5)
ax.axhline(0.95, color='r', ls='--', lw=2, label='95% threshold')
ax.axvline(min_months, color='green', ls='--', lw=2, label=f'{min_months:.1f} months')
ax.plot(min_months, 0.95, 'r*', markersize=15, zorder=5)
ax.set_xlabel('Experiment Duration [months]')
ax.set_ylabel('Discovery Probability')
ax.set_title('Discovery probability vs experiment duration')
ax.legend()
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/discovery_duration.png', dpi=200, bbox_inches='tight')
plt.close()
print("  [10/10] discovery_duration.png")

print("\nAll plots saved to plots/")
