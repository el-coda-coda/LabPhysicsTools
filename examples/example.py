import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import LabPhysicsTools as lp
from LabPhysicsTools import chi_squared_2d_map, FitPipeline, plot_typo

# ==============================================================================
# CHI_SQUARED_FIT
# ==============================================================================

# ──────────────────────────────────────────────────────────────────────────────
# ESEMPIO 1: Oscillatore smorzato ad alta frequenza
# ──────────────────────────────────────────────────────────────────────────────
def smorzata(x, A, delta, omega, phi):
    return A * np.exp(-delta * x) * np.cos(omega * x + phi)

rng = np.random.default_rng(42)
t = np.linspace(0, 4e-5, 400)
A_t, d_t, w_t, p_t = 1.5, 8e4, 5e5, 0.3
y_t = smorzata(t, A_t, d_t, w_t, p_t)
s_y = 0.05 * np.max(np.abs(y_t))
s_x = 2e-8
y_n = y_t + rng.normal(0, s_y, len(t))
x_n = t + rng.normal(0, s_x, len(t))

pip1 = FitPipeline(
    model=smorzata,
    param_names=["A", "delta", "omega", "phi"],
    p0=[1.0, 1e5, 4e5, 0.0],
    limits={"A": (0, None), "delta": (0, None)},
    run_minos=True,
    verbose=True,
)

res1 = pip1.fit(x_n, y_n, err_x=np.full_like(t, s_x), err_y=np.full_like(t, s_y))

fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
ax1.errorbar(t, y_n, yerr=s_y, fmt='.', alpha=0.4, label='dati')
ax1.plot(t, smorzata(t, *res1.valori()), 'r-', lw=1.5, label='fit')
ax1.set_ylabel('y'); ax1.legend()
ax2.errorbar(t, res1.residui, yerr=s_y, fmt='.', alpha=0.4)
ax2.axhline(0, color='r', ls='--'); ax2.set_ylabel('residui'); ax2.set_xlabel('t [s]')
fig1.tight_layout()

# ──────────────────────────────────────────────────────────────────────────────
# ESEMPIO 2: Confronto ODR Newton vs ODR Linearizzato
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("  ESEMPIO 2: Confronto Newton vs Linearizzato (Evidenza Bias)")
print("=" * 75)

def sinusoide(x, A, omega, phi):
    return A * np.sin(omega * x + phi)

A_true, w_true, phi_true = 2.0, 14.0, 0.5
N_pts = 120
x_exact = np.linspace(0, 1.2, N_pts)
y_exact = sinusoide(x_exact, A_true, w_true, phi_true)

sigma_x_val = 0.015
sigma_y_val = 0.15
sx_arr = np.full(N_pts, sigma_x_val)
sy_arr = np.full(N_pts, sigma_y_val)

x_data = x_exact + rng.normal(0, sigma_x_val, N_pts)
y_data = y_exact + rng.normal(0, sigma_y_val, N_pts)

pip2 = FitPipeline(
    model=sinusoide,
    param_names=["A", "omega", "phi"],
    p0=[1.8, 11.0, 0.0],  # p0 volutamente distante per testare la ricerca globale!
    limits={"omega": (5.0, 25.0), "A": (0.5, 5.0)},
    run_minos=True,
    verbose=False,
)

# 1. Metodo A (ODR Newton congelato)
res_newton = pip2.fit(x_data, y_data, err_x=sx_arr, err_y=sy_arr, odr_lin_Newton=True)

# 2. Metodo Linearizzato (Effective Variance)
res_lin = pip2.fit(x_data, y_data, err_x=sx_arr, err_y=sy_arr, odr_lin_Newton=False)

print(f"{'Parametro':<10} | {'Valore Vero':<12} | {'ODR Newton':<15} | {'Linearizzato':<15} | {'Diff (Lin - New)':<15}")
print("-" * 75)
for p_name in ["A", "omega", "phi"]:
    v_true = {"A": A_true, "omega": w_true, "phi": phi_true}[p_name]
    v_new = res_newton[p_name].valore
    v_lin = res_lin[p_name].valore
    print(f"{p_name:<10} | {v_true:<12.4f} | {v_new:<15.4f} | {v_lin:<15.4f} | {v_lin - v_new:<+15.4f}")
print("-" * 75)
print(f"{'chi2 / dof':<10} | {'-':<12} | {res_newton.chi2_rid:<15.4f} | {res_lin.chi2_rid:<15.4f} | {res_lin.chi2_rid - res_newton.chi2_rid:<+15.4f}")
print("=" * 75)

fig2, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
x_plot = np.linspace(np.min(x_data), np.max(x_data), 500)
ax_top.errorbar(x_data, y_data, xerr=sx_arr, yerr=sy_arr, fmt='.', color='black', alpha=0.35, label='Dati con errori su x e y')
ax_top.plot(x_plot, sinusoide(x_plot, *res_newton.valori()), 'b-', lw=2, label=f"ODR Newton (chi2/dof = {res_newton.chi2_rid:.2f})")
ax_top.plot(x_plot, sinusoide(x_plot, *res_lin.valori()), 'r--', lw=1.5, label=f"Linearizzato (chi2/dof = {res_lin.chi2_rid:.2f})")
ax_top.set_ylabel("y")
ax_top.set_title("Confronto: ODR Newton vs ODR Linearizzato")
ax_top.legend()

ax_bot.plot(x_plot, sinusoide(x_plot, *res_lin.valori()) - sinusoide(x_plot, *res_newton.valori()), 'm-', lw=1.5, label='Diff curve (Lin - New)')
ax_bot.axhline(0, color='gray', ls=':')
ax_bot.set_ylabel("Delta Fit")
ax_bot.set_xlabel("x")
ax_bot.legend()

# ──────────────────────────────────────────────────────────────────────────────
# ESEMPIO 3: Retta di calibrazione con errori su x e y
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("  ESEMPIO 3: Fit lineare (calibrazione strumentale)")
print("=" * 75)

def retta(x, m, q):
    return m * x + q

m_true, q_true = 2.45, 1.10
N_lin = 30
x_lin_exact = np.linspace(1.0, 10.0, N_lin)
y_lin_exact = retta(x_lin_exact, m_true, q_true)

# Incertezze strumentali su x e y
err_x_lin = np.full(N_lin, 0.25)
err_y_lin = np.full(N_lin, 0.40)

x_lin_meas = x_lin_exact + rng.normal(0, err_x_lin, N_lin)
y_lin_meas = y_lin_exact + rng.normal(0, err_y_lin, N_lin)

pip_lin = FitPipeline(
    model=retta,
    param_names=["m", "q"],
    p0=[1.0, 0.0],
    run_minos=True,
    verbose=True,
)

res_lin = pip_lin.fit(x_lin_meas, y_lin_meas, err_x=err_x_lin, err_y=err_y_lin)

# Grafico con bande di errore bidimensionali
fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
ax1.errorbar(x_lin_meas, y_lin_meas, xerr=err_x_lin, yerr=err_y_lin, fmt='o', color='black', alpha=0.6, label='Dati sperimentali')
x_line_plot = np.linspace(0.5, 10.5, 200)
ax1.plot(x_line_plot, retta(x_line_plot, *res_lin.valori()), 'b-', lw=1.5, label='Fit ODR Metodo A')
ax1.set_ylabel("Segnale misurato $y$ [V]")
ax1.set_title("Calibrazione lineare con incertezze su $x$ e $y$")
ax1.legend()

# Residui normalizzati (pulls)
# Calcolo dell'errore totale efficace per i residui
m_fit = res_lin["m"].valore
ax2.errorbar(x_lin_meas, res_lin.pulls, yerr=np.ones_like(x_lin_meas), fmt='o', color='black', alpha=0.6)
ax2.axhline(0, color='r', ls='--')
ax2.set_ylabel("Pulls [$\\sigma$]")
ax2.set_xlabel("Grandezza applicata $x$ [u.a.]")
ax2.set_xlabel("Grandezza applicata $x$ [u.a.]")
fig3.tight_layout()

# ──────────────────────────────────────────────────────────────────────────────
# ESEMPIO 4: Picco Gaussiano con Fondo
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("  ESEMPIO 4: Picco Gaussiano spettroscopico su fondo piatto")
print("=" * 75)

def gaussiana_fondo(x, N, mu, sigma, bg):
    return (N / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2) + bg

N_true, mu_true, sigma_true, bg_true = 50.0, 5.0, 0.6, 2.0
N_pts_gauss = 80
x_g = np.linspace(1.0, 9.0, N_pts_gauss)
y_g_clean = gaussiana_fondo(x_g, N_true, mu_true, sigma_true, bg_true)

err_x_g = np.full(N_pts_gauss, 0.04)
# Errore su y dipendente dal conteggio (Poisson approssimato)
err_y_g = np.sqrt(y_g_clean) * 0.4 + 0.1

x_g_meas = x_g + rng.normal(0, err_x_g, N_pts_gauss)
y_g_meas = y_g_clean + rng.normal(0, err_y_g, N_pts_gauss)

pip_gauss = FitPipeline(
    model=gaussiana_fondo,
    param_names=["N", "mu", "sigma", "bg"],
    p0=[30.0, 4.8, 0.8, 1.5],
    limits={"N": (0.0, None), "sigma": (0.05, 3.0), "bg": (0.0, None)},
    run_minos=True,
    verbose=True,
)

res_gauss = pip_gauss.fit(x_g_meas, y_g_meas, err_x=err_x_g, err_y=err_y_g)

fig4, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
ax1.errorbar(x_g_meas, y_g_meas, xerr=err_x_g, yerr=err_y_g, fmt='.', color='black', alpha=0.5, label='Conteggi')
x_dense = np.linspace(1.0, 9.0, 300)
ax1.plot(x_dense, gaussiana_fondo(x_dense, *res_gauss.valori()), 'g-', lw=1.8, label='Fit ODR Gaussiana')
ax1.axhline(res_gauss["bg"].valore, color='gray', ls=':', label='Fondo stimato')
ax1.set_ylabel("Intensità / Conteggi")
ax1.set_title("Spettro con picco gaussiano e risoluzione finita su $x$")
ax1.legend()

# Pannello residui normalizzati (Pulls standardizzati attorno a 0 con sigma = 1)
ax2.errorbar(x_g_meas, res_gauss.pulls, yerr=np.ones_like(x_g_meas), fmt='.', alpha=0.5)
ax2.axhline(0, color='r', ls='--')
ax2.axhline(+1, color='gray', ls=':', alpha=0.7)
ax2.axhline(-1, color='gray', ls=':', alpha=0.7)
ax2.set_ylabel("Pulls [$\\sigma$]")
ax2.set_xlabel("Canale energetico $x$ [keV]")
fig4.tight_layout()

# ──────────────────────────────────────────────────────────────────────────────
# ESEMPIO 5: Stress-Test e recupero automatico da p0 sballati
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("  ESEMPIO 5: Stress-Test con p0 completamente errati")
print("=" * 75)

def oscillatore_anharmonico(x, A, w, offset):
    return A * np.cos(w * x) + offset

N_stress = 100
x_st = np.linspace(0.0, 3.0, N_stress)
A_real, w_real, off_real = 3.0, 8.0, 1.5
y_st = oscillatore_anharmonico(x_st, A_real, w_real, off_real) + rng.normal(0, 0.2, N_stress)
sx_st = np.full(N_stress, 0.02)
sy_st = np.full(N_stress, 0.2)

# Parametri iniziali p0 volutamente disastrosi (frequenza tripla e ampiezza errata)
p0_sballato = [0.5, 23.0, -1.0]

print(f"Valori veri generati : A = {A_real}, w = {w_real}, offset = {off_real}")
print(f"Guess fornito (p0)   : A = {p0_sballato[0]}, w = {p0_sballato[1]}, offset = {p0_sballato[2]}")
print("Esecuzione pipeline (ci si attende l'avviso di recupero globale)...")

pip_stress = FitPipeline(
    model=oscillatore_anharmonico,
    param_names=["A", "w", "offset"],
    p0=p0_sballato,
    limits={"A": (0.1, 10.0), "w": (2.0, 30.0), "offset": (-5.0, 5.0)},
    run_minos=True,
    verbose=True,
)

res_stress = pip_stress.fit(x_st, y_st, err_x=sx_st, err_y=sy_st)

# Confronto visivo tra la curva di partenza (p0) e la curva convergente
fig5, ax = plt.subplots(figsize=(9, 5))
x_dense_st = np.linspace(0.0, 3.0, 400)
ax.errorbar(x_st, y_st, xerr=sx_st, yerr=sy_st, fmt='.', color='black', alpha=0.4, label='Dati')
ax.plot(x_dense_st, oscillatore_anharmonico(x_dense_st, *p0_sballato), 'r:', lw=1.5, label='Guess iniziale errato $p_0$')
ax.plot(x_dense_st, oscillatore_anharmonico(x_dense_st, *res_stress.valori()), 'b-', lw=2.0, label='Fit finale dopo ricerca globale')
ax.set_title("Recupero automatico del minimo da punto di partenza errato")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()

plt.show()

# ==============================================================================
# PLOT_TYPO
# ==============================================================================

plot_typo.setup_academic_style()

A_true = 4.5e4
mu_true = 105.7e3
sigma_true = 4.2e3

def gaussiana(x: np.ndarray) -> np.ndarray:
    return A_true * np.exp(-0.5 * ((x - mu_true) / sigma_true) ** 2)

rng = np.random.default_rng(42)

# ──────────────────────────────────────────────────────────────────────────────
# ESEMPIO 6: Scatter Gaussiana Singolo Pannello
# ──────────────────────────────────────────────────────────────────────────────

x_line = np.linspace(90e3, 120e3, 500)
y_line = gaussiana(x_line)

x_data = np.linspace(93e3, 118e3, 18)
err_y = np.full_like(x_data, 0.05 * A_true)
err_x = np.full_like(x_data, 250.0)
y_data = gaussiana(x_data) + rng.normal(0, err_y)

fig, ax = plt.subplots(figsize=(5.8, 4.4))

ax.errorbar(
    x_data, y_data, xerr=err_x, yerr=err_y,
    fmt="o", markersize=3.2, color="black", ecolor="black",
    elinewidth=0.8, capsize=0, capthick=0.8,
    label=r"Conteggi rilevati", zorder=3
)
ax.plot(x_line, y_line, color="#1f77b4", lw=1.4, label=r"Curva Gaussiana teorica", zorder=2)
ax.axvline(mu_true, color="red", ls=":", lw=1.1, label=r"$\mu_0 = 105.7\,\mathrm{kHz}$", zorder=1)

# 1. Configura prima tacche, scale e griglie
plot_typo.apply_academic_ticks_and_grid(ax, scale_x="linear", scale_y="linear")

# 2. Poi formatta le etichette e assegna il FuncFormatter
ax.set_xlabel(plot_typo.format_sci_axis(ax, x_data, r"Frequenza di scansione", axis="x", unit="Hz"))
ax.set_ylabel(plot_typo.format_sci_axis(ax, y_line, r"Intensit\`a di conteggio", axis="y", unit="u.a."))

ax.legend(loc="upper right", frameon=True, fancybox=False, facecolor="white", edgecolor="black", framealpha=1.0, borderpad=0.4, handlelength=1.4)
fig.tight_layout(pad=0.3)

# ──────────────────────────────────────────────────────────────────────────────
# ESEMPIO 7: Gaussiana + Pannello Residui
# ──────────────────────────────────────────────────────────────────────────────

x_curve = np.linspace(90e3, 122e3, 500)
y_curve = gaussiana(x_curve)

x_data2 = np.linspace(92e3, 120e3, 25)
err_y2 = np.full_like(x_data2, 0.04 * A_true)
err_x2 = np.full_like(x_data2, 200.0)
y_data2 = gaussiana(x_data2) + rng.normal(0, err_y2)
residui = y_data2 - gaussiana(x_data2)

fig = plt.figure(figsize=(6.0, 5.2))
gs = gridspec.GridSpec(2, 1, height_ratios=[3.2, 1.0])
gs.update(hspace=0.0)

ax_main = fig.add_subplot(gs[0, 0])
ax_res = fig.add_subplot(gs[1, 0], sharex=ax_main)

# Pannello superiore
ax_main.errorbar(x_data2, y_data2, xerr=err_x2, yerr=err_y2, fmt="o", markersize=3.0, color="black", ecolor="black", elinewidth=0.8, capsize=0, capthick=0.8, label=r"Dati sperimentali", zorder=3)
ax_main.plot(x_curve, y_curve, color="#1f77b4", lw=1.4, label=r"Curva teorica", zorder=2)
ax_main.axvline(mu_true, color="red", ls=":", lw=1.1, label=r"$\mu_0 = 105.7\,\mathrm{kHz}$", zorder=1)

# Pannello inferiore
ax_res.errorbar(x_data2, residui, yerr=err_y2, fmt="o", markersize=3.0, color="black", ecolor="black", elinewidth=0.8, capsize=0, capthick=0.8, zorder=3)
ax_res.axhline(0.0, color="red", ls="--", lw=1.1, zorder=2)
ax_res.axvline(mu_true, color="red", ls=":", lw=1.1, zorder=1)

# 1. Configura tacche e griglie prima di formattare
plot_typo.apply_academic_ticks_and_grid(ax_main, scale_x="linear", scale_y="linear")
plot_typo.apply_academic_ticks_and_grid(ax_res, scale_x="linear", scale_y="linear")
ax_main.tick_params(axis="x", labelbottom=False)

# 2. Impostazione range e locator per i residui (simmetrico e pulito)
ax_res.set_yticks([-5e3, 0.0, 5e3])
ax_res.set_ylim(-7.5e3, 7.5e3)
ax_main.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune="lower"))  # Toglie lo 0.00 che tocca la riga!

# 3. Formattatori scientifici (chiamati per ULTIMI)
ax_main.set_ylabel(plot_typo.format_sci_axis(ax_main, y_curve, r"Intensit\`a", axis="y", unit="u.a."))
ax_res.set_ylabel(plot_typo.format_sci_axis(ax_res, residui, r"Residui", axis="y", unit="u.a.", dec=1))
ax_res.set_xlabel(plot_typo.format_sci_axis(ax_res, x_data2, r"Frequenza di scansione", axis="x", unit="Hz"))

ax_main.legend(loc="upper right", frameon=True, fancybox=False, facecolor="white", edgecolor="black", framealpha=1.0, borderpad=0.4, handlelength=1.4)
fig.tight_layout(pad=0.25)

# ──────────────────────────────────────────────────────────────────────────────
# ESEMPIO 8: Diagramma di Bode Modulo + Residui (Semilogaritmico)
# ──────────────────────────────────────────────────────────────────────────────

f0 = 10.0e3
Q = 3.0

def modulo_db(f: np.ndarray) -> np.ndarray:
    s = 1j * (f / f0)
    H = 1.0 / (1.0 + (s / Q) + s**2)
    return 20.0 * np.log10(np.abs(H))

f_dense = np.logspace(3, 5, 500)
mag_dense = modulo_db(f_dense)

f_data3 = np.logspace(3.1, 4.9, 22)
err_mag = np.full_like(f_data3, 0.45)
mag_data = modulo_db(f_data3) + rng.normal(0, err_mag)
residui_bode = mag_data - modulo_db(f_data3)

fig = plt.figure(figsize=(6.0, 5.2))
gs = gridspec.GridSpec(2, 1, height_ratios=[3.2, 1.0])
gs.update(hspace=0.0)

ax_main = fig.add_subplot(gs[0, 0])
ax_res = fig.add_subplot(gs[1, 0], sharex=ax_main)

ax_main.plot(f_dense, mag_dense, color="#1f77b4", lw=1.4, label=r"Modello $|H(f)|$", zorder=2)
ax_main.errorbar(f_data3, mag_data, yerr=err_mag, fmt="o", markersize=3.0, color="black", ecolor="black", elinewidth=0.8, capsize=1.6, capthick=0.8, label=r"Dati misurati", zorder=3)
ax_main.axvline(f0, color="red", ls=":", lw=1.1, label=r"$f_0 = 10\,\mathrm{kHz}$", zorder=1)

ax_res.errorbar(f_data3, residui_bode, yerr=err_mag, fmt="o", markersize=3.0, color="black", ecolor="black", elinewidth=0.8, capsize=1.6, capthick=0.8, zorder=3)
ax_res.axhline(0.0, color="red", ls="--", lw=1.1, zorder=2)
ax_res.axvline(f0, color="red", ls=":", lw=1.1, zorder=1)

plot_typo.apply_academic_ticks_and_grid(ax_main, scale_x="log", scale_y="linear")
plot_typo.apply_academic_ticks_and_grid(ax_res, scale_x="log", scale_y="linear")
ax_main.tick_params(axis="x", labelbottom=False)

max_dev = np.max(np.abs(residui_bode) + err_mag)
ax_res.set_ylim(-1.35 * max_dev, 1.35 * max_dev)
ax_res.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, symmetric=True, prune="both"))

ax_main.set_ylabel(r"Ampiezza $[\mathrm{dB}]$")
ax_res.set_xlabel(r"Frequenza $[\mathrm{Hz}]$")
ax_res.set_ylabel(r"Residui $[\mathrm{dB}]$")

ax_main.legend(loc="lower left", frameon=True, fancybox=False, facecolor="white", edgecolor="black", framealpha=1.0, borderpad=0.4, handlelength=1.4)
fig.tight_layout(pad=0.25)

# ──────────────────────────────────────────────────────────────────────────────
# ESEMPIO 9: Grafico Log-Log Puro
# ──────────────────────────────────────────────────────────────────────────────

A_theo = 3.2e-4
alpha = -1.65

def power_law(f: np.ndarray) -> np.ndarray:
    return A_theo * (f**alpha)

f_dense = np.logspace(1, 4, 400)
S_dense = power_law(f_dense)

f_data4 = np.logspace(1.2, 3.8, 16)
S_true = power_law(f_data4)
err_S = 0.12 * S_true
S_data = S_true + rng.normal(0, err_S)

fig, ax = plt.subplots(figsize=(5.8, 4.4))

ax.plot(f_dense, S_dense, color="#2ca02c", lw=1.4, label=rf"Modello $S(f) \propto f^{{{alpha:.2f}}}$", zorder=2)
ax.errorbar(f_data4, S_data, yerr=err_S, fmt="o", markersize=3.2, color="black", ecolor="black", elinewidth=0.8, capsize=1.8, capthick=0.8, label=r"Dati spettrali", zorder=3)

plot_typo.apply_academic_ticks_and_grid(ax, scale_x="log", scale_y="log")

ax.set_xlabel(r"Frequenza $[\mathrm{Hz}]$")
ax.set_ylabel(r"Densit\`a spettrale \, $[\mathrm{V^2 / Hz}]$")

ax.legend(loc="upper right", frameon=True, fancybox=False, facecolor="white", edgecolor="black", framealpha=1.0, borderpad=0.4, handlelength=1.4)
fig.tight_layout(pad=0.3)

plt.show()

# ==============================================================================
# CHI_SQUARED_2D_MAP
# ==============================================================================

# ──────────────────────────────────────────────────────────────────────────────
# ESEMPIO 10: Mappa 2D del chi quadro
# ──────────────────────────────────────────────────────────────────────────────

def modello_lineare(x, m, q):
    return m * x + q

rng = np.random.default_rng(123)
m_vero, q_vero = 2.45, 1.10
n_punti = 35

x_pulito = np.linspace(1.0, 10.0, n_punti)
y_pulito = modello_lineare(x_pulito, m_vero, q_vero)
err_x = np.full(n_punti, 0.20)
err_y = np.full(n_punti, 0.45)

x_misurato = x_pulito + rng.normal(0, err_x, n_punti)
y_misurato = y_pulito + rng.normal(0, err_y, n_punti)

pipeline = FitPipeline(model=modello_lineare, param_names=["m", "q"], p0=[1.0, 0.0], run_minos=True, verbose=True)
risultati_fit = pipeline.fit(x=x_misurato, y=y_misurato, err_x=err_x, err_y=err_y, odr_lin_Newton=True)

info_parametri = {
    "m": {"label": r"$m$", "unit": r"V/s"},
    "q": {"label": r"$q$", "unit": r"V"},
}

plot_typo.setup_academic_style(textsize=12)

fig, axes = chi_squared_2d_map.chi2_map_pipeline(
    risultati=risultati_fit,
    param_x="m",
    param_y="q",
    param_info=info_parametri,
    figsize=(7.5, 6.5),
    stats_box_loc="custom",
    stats_box_pos=(0.725, 0.85),
    show=True
)