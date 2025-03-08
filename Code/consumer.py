import numpy as np
from HARK.distributions.multivariate import MultivariateLogNormal as MVLogNormal
from scipy.optimize import root_scalar
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import bernoulli, lognorm
from matplotlib import pyplot as plt

class BufferStockModel:

    def __init__(self, ρ = 4, Γ = 1.03, ℜ = 1.04, Rf = 1.01, β = 0.93, Σ = np.array([[0.01, 0.0, 0.0], [0.0, 0.0225, 0.0], [0.0, 0.0, 0.01]])):

        self.ρ, self.Γ, self.ℜ, self.Rf, self.β, self.Σ = ρ, Γ, ℜ, Rf, β, Σ

        self.μ = - np.diag(Σ) / 2
        
        self.shocks = MVLogNormal(self.μ, self.Σ)

    def u(self, c):

        ρ = self.ρ

        if ρ == 1:
            return np.log(c)
        elif ρ >= 0:
            return (c ** (1 - ρ)) / (1 - ρ)
        else:
            raise ValueError("ρ must be non-negative")

    def u_prime(self, c):

        ρ = self.ρ

        if ρ < 0:
            raise ValueError("ρ must be non-negative")
        elif ρ == 0:
            return 1
        else:
            return c ** (-ρ)

    def m1(self, a, ς, atom):

        ψ, ν, ζ = atom

        ℜ, Γ, Rf, ℘ = self.ℜ, self.Γ, self.Rf, self.℘

        R_eff = Rf + ς * (ν * ℜ - Rf)

        G_eff = Γ * ψ

        mNrm = R_eff * a / G_eff + ζ / (1 - ℘)

        return mNrm

    def m2(self, a, ς, atom):

        ψ, ν, ζ = atom

        ℜ, Γ, Rf = self.ℜ, self.Γ, self.Rf

        R_eff = Rf + ς * (ν * ℜ - Rf)

        G_eff = Γ * ψ

        mNrm = R_eff * a / G_eff

        return mNrm

    def ς_euler(self, ς, a, c_next):

        Rf, ℜ, Γ, ρ, ℘, shocks_approx = self.Rf, self.ℜ, self.Γ, self.ρ, self.℘, self.shocks_approx

        def ℓ1(atom):
            return self.m1(a, ς, atom)

        def ℓ2(atom):
            return self.m2(a, ς, atom)

        def g(atom):

            ψ, ν, ζ = atom

            return ((ℜ * ν) - Rf) * ((Γ * ψ) ** (-ρ)) * ((1 - ℘) * self.u_prime(c_next(ℓ1(atom))) + ℘ * self.u_prime(c_next(ℓ2(atom))))

        return shocks_approx.expected(func = g)

    def ς_hat(self, a, c_next):

        shocks_approx = self.shocks_approx

        ς_func = lambda ς : self.ς_euler(ς, a, c_next)

        ς = root_scalar(ς_func, method="newton", x0 = 0.5)['root']

        return ς

    def c_hat(self, a, ς, c_next):

        β, Rf, ℜ, Γ, ρ, ℘, shocks_approx = self.β, self.Rf, self.ℜ, self.Γ, self.ρ, self.℘, self.shocks_approx

        def ℓ1(atom):

            return self.m1(a, ς, atom)

        def ℓ2(atom):

            return self.m2(a, ς, atom)

        def c_rho(atom):

            ψ, ν, ζ = atom

            return β * (Rf + ς * (ℜ * ν - Rf)) * ((Γ * ψ) ** (-ρ)) * ((1 - ℘) * self.u_prime(c_next(ℓ1(atom))) + ℘ * self.u_prime(c_next(ℓ2(atom))))

        return shocks_approx.expected(func = c_rho) ** (-1/ρ)

    def discretize_distribution(self, N, tail_bound = [0.0015, 0.9985], decomp="cholesky", endpoints = True):

        self.shocks_approx = self.shocks._approx_equiprobable(N = N, tail_bound = tail_bound, decomp=decomp, endpoints=endpoints)

    def gen_asset_grid(self, aMax, n = 20, boroConst = True, artBoroConst = 0, ℘ = 0.005):

        worst = self.shocks_approx.atoms.min(axis=1)

        ψMin, νMin, ζMin = worst

        Γ, ℜ = self.Γ, self.ℜ

        if boroConst:
            if artBoroConst == 0.0:
                aMin = 1e-5
            else:
                aMin = artBoroConst

            self.℘ = 0
        else:
            aMin = 1e-5
            self.℘ = ℘

        aDiffs = np.exp(np.linspace(-4, np.log(aMax - aMin), n-1))

        aNrmGrid = np.empty(n)

        aNrmGrid[0] = aMin
        aNrmGrid[1:] = aMin + aDiffs

        self.aNrmGrid = aNrmGrid
        self.boroConst = boroConst

    def pre_solve(self, N = 10, aMax = 6, tail_bound = [0.0015, 0.9985], decomp="cholesky", endpoints = True, n = 20, boroConst = True, artBoroConst = 0, ℘ = 0.005):

        self.discretize_distribution(N = N, tail_bound = tail_bound, decomp=decomp, endpoints=endpoints)

        self.gen_asset_grid(aMax = aMax, n = n, boroConst = boroConst, artBoroConst = artBoroConst, ℘ = ℘)

    def solve(self, T = None, c_guess = lambda x : x, interp="linear", max_iter = 100, tol=1e-6):

        boroConst = self.boroConst
        aNrmGrid = self.aNrmGrid
        n = len(aNrmGrid)

        if T is None:
            c_T = c_guess
            t = 0
            iter = 0
            error = tol + 1
        else:
            t = T
            c_T = lambda x : x
            iter = max_iter + 1
            error = tol - 1

        policies = [{'cFunc' : c_T}]

        while (t > 0) or ((T is None) & (iter < max_iter) & (error > tol)):
            c_next = policies[-1]['cFunc']
            
            ς_hat_vec = np.vectorize(lambda a : self.ς_hat(a, c_next))

            ς_opt = ς_hat_vec(aNrmGrid)

            ςGrid = np.where(ς_opt > 1, 1, np.where(ς_opt < 0, 0, ς_opt))

            def c_hat_opt(a, ς):

                return self.c_hat(a, ς, c_next)

            c_hat_vec = np.vectorize(c_hat_opt)

            cNrmGrid = c_hat_vec(aNrmGrid, ςGrid)

            mNrmGrid = cNrmGrid + aNrmGrid

            if boroConst:
                ςvals = np.empty(n+1)
                ςvals[0] = ςGrid[0]
                ςvals[1:] = ςGrid

                cvals = np.empty(n+1)
                cvals[0] = 0.0
                cvals[1:] = cNrmGrid

                mvals = np.zeros(n+1)
                mvals[1:] = mNrmGrid
            else:
                ςvals = ςGrid
                cvals = cNrmGrid
                mvals = mNrmGrid

            if interp=="linear":
                cFunc_t = InterpolatedUnivariateSpline(mvals, cvals, k=1)
                ςFunc_t = InterpolatedUnivariateSpline(mvals, ςvals, k=1)
            elif interp=="spline":
                cFunc_t = InterpolatedUnivariateSpline(mvals, cvals, k=3)
                ςFunc_t = InterpolatedUnivariateSpline(mvals, ςvals, k=3)
            elif interp=="SL":
                cFunc_t = InterpolatedUnivariateSpline(mvals, cvals, k=3)
                ςFunc_t = InterpolatedUnivariateSpline(mvals, ςvals, k=1)

            policies_t = {"cFunc":cFunc_t, "ςFunc":ςFunc_t, "mNrmGrid":mvals, "ςGrid":ςvals, "cNrmGrid":cvals}

            policies.append(policies_t)

            if T is not None:
                t -= 1
            else:
                iter += 1

                if iter > 1:
                    m_prev = policies[-2]['mNrmGrid']
                    ς_prev = policies[-2]['ςGrid']
                    errors = np.array([np.max(np.abs(mvals - m_prev)), np.max(np.abs(ςvals - ς_prev))])
                    error = np.max(errors)

        policies.reverse()

        params = {}

        if T is None:
            params['T'] = 'Infinite'
            params['error'] = error
            params['iter'] = iter
        else:
            params['T'] = T

        params['interp'] = interp

        self.policies = policies
        self.params = params

        self.gen_solution()

    def gen_solution(self):

        T = self.params['T']

        if T == 'Infinite':
            self.solution = self.policies[0]
        else:

            def solution(t : int):

                if t < 0:
                    raise ValueError("t must be non-negative")
                elif t > T:
                    raise ValueError("t must be between 0 and T")
                else:
                    return self.policies[t]

            self.solution = solution

    def plot_cFunc(self, path=None, mGrid = np.linspace(0, 10, 300), t = None):

        if (self.params['T'] == 'Infinite') and (t is not None):
            raise ValueError('The computed solution is for the infinite horizon model')
        elif (self.params['T'] != 'Infinite') and (t is None):
            raise ValueError('The solution to a finite horizon model must be provided for a given t')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if t is None:
            ax.plot(mGrid, self.solution['cFunc'](mGrid), color = "black", linestyle="dashed", label = "$c(m)$")

            ax.legend()
        else:
            ax.plot(mGrid, self.solution(t)['cFunc'](mGrid), color = "black", linestyle = "dashed", label = f'$c_{t}(m)$')

            ax.legend()
        
        ax.set_xlabel('$m$')
        ax.set_ylabel('$c$')

        if path is not None:
            fig.savefig(path)
            plt.clf()
        else:
            fig.show()

    def plot_ςFunc(self, path=None, mGrid = np.linspace(0, 10, 300), t = None, ms_line = False):

        s_nu, Rf, ℜ, ρ = self.Σ[1, 1], self.Rf, self.ℜ, self.ρ

        if (self.params['T'] == 'Infinite') and (t is not None):
            raise ValueError('The computed solution is for the infinite horizon model')
        elif (self.params['T'] != 'Infinite') and (t is None):
            raise ValueError('The solution to a finite horizon model must be provided for a given t')
        
        fig, ax = plt.subplots(figsize=(8, 6))

        if t is None:
            ax.set_ylim((0, 1.1))
            ax.plot(mGrid, self.solution['ςFunc'](mGrid), color = "black", label = "$\\varsigma(m)$")
        else:
            ax.set_ylim((0, 1.1))
            ax.plot(mGrid, self.solution(t)['ςFunc'](mGrid), color = "black", label = f"$\\varsigma_{t+1}(m)$")

        if ms_line:

            ax.axhline(y = (ℜ - Rf) / (ρ * s_nu), color = "orange", linestyle = "dashed", label="$\\varsigma^*$ (M-S)")

        ax.legend()

        ax.set_xlabel('$m$')
        ax.set_ylabel('$\\varsigma$')

        if path is not None:
            fig.savefig(path)
            plt.clf()
        else:
            fig.show()

    def target_wealth(self, mGrid = np.linspace(0, 10, 300), t=None):
        if (self.params['T'] == 'Infinite') and (t is not None):
            raise ValueError('The computed solution is for the infinite horizon model')
        elif (self.params['T'] != 'Infinite') and (t is None):
            raise ValueError('The solution to a finite horizon model must be provided for a given t')

        if t is None:

            cFunc = self.solution['cFunc']
            ςFunc = self.solution['ςFunc']

            ℜ, Γ, Rf, ℘ = self.ℜ, self.Γ, self.Rf, self.℘
            shocks_approx = self.shocks_approx

            def ℓ(m, atom):

                ψ, ν, ζ = atom

                mNext = (m - cFunc(m)) * (Rf + ςFunc(m) * (ℜ * ν - Rf)) / (Γ * ψ) + ζ

                return mNext

            def g(m):

                def f(atom):
                    return ℓ(m, atom)

                return shocks_approx.expected(func = f)

            mNext = np.empty(len(mGrid))

            for i in range(len(mNext)):
                mNext[i] = g(mGrid[i])

            mNextFunc = InterpolatedUnivariateSpline(mGrid, mNext, k=1)

            m = root_scalar(lambda m : mNextFunc(m) - m, method="newton", x0=3)['root']

            self.target_m = m

            return m

        else:

            cFunc = self.solution(t)['cFunc']
            ςFunc = self.solution(t)['ςFunc']

            ℜ, Γ, Rf, ℘ = self.ℜ, self.Γ, self.Rf, self.℘
            shocks_approx = self.shocks_approx

            def ℓ(m, atom):

                ψ, ν, ζ = atom

                mNext = (m - cFunc(m)) * (Rf + ςFunc(m) * (ℜ * ν - Rf)) / (Γ * ψ) + ζ

                return mNext

            def g(m):

                def f(atom):
                    return ℓ(m, atom)

                return shocks_approx.expected(func = f)

            mNext = np.empty(len(mGrid))

            for i in range(len(mNext)):
                mNext[i] = g(mGrid[i])

            mNextFunc = InterpolatedUnivariateSpline(mGrid, mNext, k=1)

            m = root_scalar(lambda m : mNextFunc(m) - m, method="newton", x0=3)['root']

            return m


    def plot_targetς(self, path=None, mGrid = np.linspace(0, 10, 300), t = None):
        if (self.params['T'] == 'Infinite') and (t is not None):
            raise ValueError('The computed solution is for the infinite horizon model')
        elif (self.params['T'] != 'Infinite') and (t is None):
            raise ValueError('The solution to a finite horizon model must be provided for a given t')

        if t is None:
            ςFunc = self.solution['ςFunc']

            ℜ, Γ, Rf = self.ℜ, self.Γ, self.Rf

            m = self.target_wealth()

            fig, ax = plt.subplots(figsize=(8, 6))

            ax.plot(mGrid, ςFunc(mGrid), color = "black", label = "$\\varsigma(m_{t})$")

            ax.axvline(x = m, color='black', linestyle='dashed', label="$m^*$")

            ax.legend()

            ax.set_xlabel('$m$')
            ax.set_ylabel('$\\varsigma$')

            if path is not None:
                fig.savefig(path)
                plt.clf()
            else:
                fig.show()


    def plot_wealth_transition(self, path=None, mGrid = np.linspace(0, 5, 300), t=None):
        if (self.params['T'] == 'Infinite') and (t is not None):
            raise ValueError('The computed solution is for the infinite horizon model')
        elif (self.params['T'] != 'Infinite') and (t is None):
            raise ValueError('The solution to a finite horizon model must be provided for a given t')

        if t is None:

            cFunc = self.solution['cFunc']
            ςFunc = self.solution['ςFunc']

            ℜ, Γ, Rf, ℘ = self.ℜ, self.Γ, self.Rf, self.℘
            shocks_approx = self.shocks_approx

            def ℓ(m, atom):

                ψ, ν, ζ = atom

                mNext = (m - cFunc(m)) * (Rf + ςFunc(m) * (ℜ * ν - Rf)) / (Γ * ψ) + ζ

                return mNext

            def g(m):

                def f(atom):
                    return ℓ(m, atom)

                return shocks_approx.expected(func = f)

            mNext = np.empty(len(mGrid))

            for i in range(len(mNext)):
                mNext[i] = g(mGrid[i])

            fig, ax = plt.subplots(figsize=(8, 6))

            ax.plot(mGrid, mGrid, linestyle="dashed", color="black", label="$m_t$")
            ax.plot(mGrid, mNext, color="black", label="$\mathbb{E}_t[m_{t+1}]$")
            ax.legend()

            if path is not None:
                fig.savefig(path)
                plt.clf()
            else:
                fig.show()

    def gen_cond_distribution(self, ν):
        
        Σ = self.Σ
        μ = self.μ

        ln_ν = np.log(ν)

        Σ_11 = Σ[0:3:2, 0:3:2]

        Σ_12 = Σ[0:3:2, 1]

        Σ_22 = Σ[1, 1]

        Σ_c = Σ_11 - (Σ_12 @ Σ_12.T) / Σ_22

        μ_c = μ[0:3:2] + Σ_12 * (ln_ν - μ[1]) / Σ_22

        return MVLogNormal(μ_c, Σ_c)

    def simulate_inf_horizon(self, N=10000, T=120):
        
        ℜ, Γ, Rf, ℘ = self.ℜ, self.Γ, self.Rf, self.℘

        Σ = self.Σ
        μ = self.μ

        ν_dist = lognorm(s=np.sqrt(Σ[1, 1]), scale=1)

        ℘_dist = bernoulli(p=1 - ℘)

        cFunc = self.solution['cFunc']
        ςFunc = self.solution['ςFunc']

        mDist = np.ones(N)

        for _ in range(T):

            ςDist = ςFunc(mDist)
            
            ν = ν_dist.rvs()

            cond_dist = self.gen_cond_distribution(ν)

            shocks = cond_dist.rvs(N).T

            z_inc = ℘_dist.rvs(N)

            aDist = mDist - cFunc(mDist)

            mNew = (Rf + ςDist * (ℜ * ν - Rf)) * aDist / (Γ * shocks[0, :]) + z_inc * shocks[1, :] / (1 - ℘)

            mDist = mNew

        values = np.empty((N, 2))

        values[:, 0] = mDist
        values[:, 1] = ςDist

        return values