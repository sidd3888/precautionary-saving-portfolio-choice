from consumer import BufferStockModel
import seaborn as sns, pandas as pd, numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import rc

# Set the working directory

os.chdir(os.path.dirname(__file__), )
os.chdir('../Figures/')

mGrid = np.linspace(0, 300, 300)

def gen_cov_matrix(ω : float, kind = "P"):

    if kind == "P":
        Σ = np.array(
            [
                [0.01, ω, 0.0],
                [ω, 0.0225, 0.0],
                [0.0, 0.0, 0.01]
            ]
        )
    elif kind == "T":
        Σ = np.array(
            [
                [0.01, 0.0, 0.0],
                [0.0, 0.0225, ω],
                [0.0, ω, 0.01]
            ]
        )

    return Σ

def gen_us_cov_matrix(ω : float, kind = "P"):

    if kind == "P":
        Σ = np.array(
            [
                [0.01, ω, 0.0],
                [ω, 0.011, 0.0],
                [0.0, 0.0, 0.01]
            ]
        )
    elif kind == "T":
        Σ = np.array(
            [
                [0.01, 0.0, 0.0],
                [0.0, 0.011, ω],
                [0.0, ω, 0.01]
            ]
        )

    return Σ

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 16})
rc('text', usetex=True)

# Baseline NBC model

consumer = BufferStockModel()

consumer.pre_solve(aMax = 300, n=70, boroConst=True, N = 12)

consumer.solve()

consumer.plot_ςFunc(path = 'portfolio_baseline.png', mGrid = mGrid, ms_line=True)

consumer.plot_cFunc(path = 'NBCcons_baseline.png')

# NBC model with permanent income correlation

Σ = gen_cov_matrix(0.008)

consumer = BufferStockModel(Σ = Σ)

consumer.pre_solve(aMax = 300, n= 70, boroConst = True, N=12)

consumer.solve()

consumer.plot_ςFunc(path = 'NBCportfolio_permanent_low_income.png')

consumer.plot_ςFunc(path = 'NBCportfolio_permanent_high_income.png', mGrid = mGrid, ms_line = True)

consumer.plot_targetς(path = 'NBCshare_target.png', mGrid=np.linspace(0, 4, 100))

values = consumer.simulate_inf_horizon()
values_df = pd.DataFrame(values, columns=["$m$", "$\\varsigma$"])

plt.rcParams["figure.figsize"] = (8,6)

sns.histplot(values_df, x='$m$', kde=True).get_figure().savefig('NBC_mdist.png')

plt.clf()

plt.rcParams["figure.figsize"] = (8,6)

sns.histplot(values_df, x='$\\varsigma$', kde=True).get_figure().savefig('NBC_sharedist.png')

plt.clf()

plt.rcParams["figure.figsize"] = (8,6)

sns.displot(values_df, x='$m$', y='$\\varsigma$').savefig('NBC_jointdist.png')

plt.clf()

# NBC model with transitory income correlation

Σ = gen_cov_matrix(0.014, kind="T")

consumer = BufferStockModel(Σ = Σ)

consumer.pre_solve(aMax = 300, n= 70, boroConst = True, N=12)

consumer.solve()

consumer.plot_ςFunc(path = 'NBCportfolio_transitory_low_income.png')

consumer.plot_ςFunc(path = 'NBCportfolio_transitory_high_income.png', mGrid = mGrid, ms_line = True)

# Solve the two-period version of the previous model

Σ = gen_cov_matrix(0.008, kind="T")

consumer = BufferStockModel(Σ = Σ)

consumer.pre_solve(aMax = 300, n= 70, boroConst = True, N=12)

consumer.solve(T=1)

consumer.plot_ςFunc(t=0, path = 'NBC_lastprd.png')

# NIR model with permanent income correlation

Σ = gen_cov_matrix(0.008)

consumer = BufferStockModel(Σ = Σ)

consumer.pre_solve(aMax = 300, n= 70, boroConst = False, N=12)

consumer.solve()

consumer.plot_ςFunc(path = 'NIRportfolio_permanent_low_income.png')

consumer.plot_targetς(path = 'NIRshare_target', mGrid = np.linspace(0, 4, 100))

values = consumer.simulate_inf_horizon()
values_df = pd.DataFrame(values, columns=["$m$", "$\\varsigma$"])

plt.rcParams["figure.figsize"] = (8,6)

sns.histplot(values_df, x='$m$', kde=True).get_figure().savefig('NIR_mdist.png')

plt.clf()

plt.rcParams["figure.figsize"] = (8,6)

sns.histplot(values_df, x='$\\varsigma$', kde=True).get_figure().savefig('NIR_sharedist.png')

plt.clf()

plt.rcParams["figure.figsize"] = (8,6)

sns.displot(values_df, x='$m$', y='$\\varsigma$').savefig('NIR_jointdist.png')

plt.clf()

# Portfolio share limit in the NBC model

Ω = [0.0025 * (i + 1) for i in range(5)]

consumers = []

for i in range(len(Ω)):
    
    Σ = gen_cov_matrix(Ω[i])
    
    consumer = BufferStockModel(Σ = Σ)
    
    consumer.pre_solve(aMax = 1000, n = 80, boroConst = True, N=12)
    
    consumer.solve()

    consumers.append(consumer)

fig, ax = plt.subplots(figsize=(8, 6))

for i in range(len(consumers)):
    mGrid2 = np.linspace(0, 1000, 1000)

    ax.plot(mGrid2, consumers[i].solution['ςFunc'](mGrid2), label = f"$\omega = {Ω[i]}$")

ax.axhline(y = (1.04 - 1.01) / (4 * 0.0225), color = "black", linestyle = "dashed", label="$\\varsigma^*$ (M-S)")

ax.set_xlabel('$m$')
ax.set_ylabel('$\\varsigma$')

ax.legend()
fig.savefig('NBCshare_limit.png')

plt.clf()

# Portfolio share limit in the NIR model

Ω = [0.0025 * (i + 1) for i in range(5)]

consumers_z = []

for i in range(len(Ω)):
    
    Σ = gen_cov_matrix(Ω[i])
    
    consumer = BufferStockModel(Σ = Σ)
    
    consumer.pre_solve(aMax = 1000, n = 80, boroConst = False, N=12)
    
    consumer.solve()

    consumers_z.append(consumer)

fig, ax = plt.subplots(figsize=(8, 6))

for i in range(len(consumers)):
    mGrid2 = np.linspace(0, 1000, 1000)

    ax.plot(mGrid2, consumers_z[i].solution['ςFunc'](mGrid2), label = f"$\omega = {Ω[i]}$")

ax.axhline(y = (1.04 - 1.01) / (4 * 0.0225), color = "black", linestyle = "dashed", label="$\\varsigma^*$ (M-S)")

ax.set_xlabel('$m$')
ax.set_ylabel('$\\varsigma$')

ax.legend()
fig.savefig('NIRshare_limit.png')

plt.clf()

# NBC model calibrated to US data

Σ = gen_us_cov_matrix(0.01)

consumer = BufferStockModel(Σ = Σ, ℜ = 1.0767, Rf = 1.0131)

consumer.pre_solve(aMax = 300, n = 70, boroConst = True, N=12)

consumer.solve()

consumer.plot_ςFunc(path = 'NBCcalibrated_RRA4.png')

# NBC model calibrated to US data with ρ=12

Σ = gen_us_cov_matrix(0.01)

consumer = BufferStockModel(Σ = Σ, ℜ = 1.0767, Rf = 1.0131, ρ=12)

consumer.pre_solve(aMax = 300, n = 70, boroConst = True, N=12)

consumer.solve()

consumer.plot_ςFunc(path = 'NBCcalibrated_RRA12.png', mGrid=mGrid)

# NBC model calibrated to US data with ρ=7

Σ = gen_us_cov_matrix(0.01)

consumer = BufferStockModel(Σ = Σ, ℜ = 1.0767, Rf = 1.0131, ρ=7)

consumer.pre_solve(aMax = 300, n = 70, boroConst = True, N=12)

consumer.solve()

consumer.plot_targetς(path = 'NBCcalibrated_target.png', mGrid=np.linspace(0, 4, 100))

# NIR model calibrated to US data with ρ=7.5

Σ = gen_us_cov_matrix(0.01)

consumer = BufferStockModel(Σ = Σ, ℜ = 1.0767, Rf = 1.0131, ρ = 7.5)

consumer.pre_solve(aMax = 300, n = 70, boroConst = False, N=12)

consumer.solve()

consumer.plot_targetς(path = 'NIRcalibrated_target.png', mGrid=np.linspace(0, 4, 100))
