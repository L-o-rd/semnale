from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp

if __name__ == '__main__':
    samples = pd.read_csv('./labs/10/co2_daily_mlo.csv', comment='#', header=None,
                       names=["Year", "Month", "Day", "DecimalDate", "PPM"],
                       na_values=-99.99)
    samples = samples.dropna(subset=["PPM"])
    
    # (a) seria lunara 
    
    samples['Date'] = pd.to_datetime(samples[['Year', 'Month', 'Day']])
    samples['Monthly'] = samples['Date'].dt.to_period('M')
    monthly = samples.groupby('Monthly')['PPM'].mean().reset_index()
    monthly['Monthly'] = monthly['Monthly'].dt.to_timestamp()
    monthly.rename(columns={'Monthly': 'Month', 'PPM': 'PPM'}, inplace=True)

    # (b) eliminarea trendului
    # b.0 eliminare prin scadere

    detrended = np.zeros(monthly.shape[0])
    for i in range(1, len(monthly['PPM'].values)):
        detrended[i] = monthly['PPM'].values[i] - monthly['PPM'].values[i - 1]

    monthly['SPPM'] = detrended

    # b.1 eliminare prin regresie

    X = np.arange(len(monthly)).reshape(-1, 1)  
    y = monthly['PPM'].values      

    lmodel = LinearRegression()
    lmodel.fit(X, y)

    trend = lmodel.predict(X)
    detrended = y - trend
    monthly['DPPM'] = detrended

    # (c) regresie cu proces gaussian

    last12 = monthly.iloc[-12:]
    X = np.arange(len(last12)).reshape(-1, 1)
    y = last12['DPPM'].values

    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
    gp.fit(X, y)

    Xp = np.linspace(0, 11, 100).reshape(-1, 1) 
    yp, ystd = gp.predict(Xp, return_std=True)
    
    _, axs = plt.subplots(4, 1, figsize = (10, 6))
    axs[0].plot(monthly['Month'], monthly['PPM'])
    axs[1].plot(monthly['Month'], monthly['SPPM'])
    axs[2].plot(monthly['Month'], monthly['DPPM'])
    axs[3].plot(X, y, 'o', label = 'Detrended (last 12 months)')
    axs[3].plot(Xp, yp, 'r-', label = 'Prediction (last 12 months)')
    axs[3].fill_between(Xp.ravel(), yp - ystd, yp + ystd, color='r', alpha=0.2, label="Confidence Interval")
    axs[3].legend()
    plt.tight_layout()
    plt.savefig('./labs/10/03a.pdf')
    plt.show()

    def rbf(x, y, l = 1.0, s = 1.0):
        d = np.sum(x ** 2, axis = 1).reshape(-1, 1) + np.sum(y ** 2, 1) - 2 * np.dot(x, y.T)
        return s * np.exp(-0.5 * (1.0 / l) * d)

    l = -12
    A = monthly.iloc[l:]

    Xb = np.linspace(0, len(A) - 1, 10 ** 2).reshape(-1, 1)
    Xa = np.arange(len(A)).reshape(-1, 1)
    ya = A['DPPM'].values  

    Caa = rbf(Xa, Xa, 0.3)
    Cab = rbf(Xa, Xb, 0.3)
    Cbb = rbf(Xb, Xb, 0.3)
    sol = sp.linalg.solve(Caa, Cab, assume_a = 'pos').T

    m = sol @ ya
    D = Cbb - (sol @ Cab)
    s = np.sqrt(np.abs(np.diag(D)))
    
    samples = np.random.multivariate_normal(m.flatten(), D, size = 1)
    plt.figure(figsize = (10, 6))
    plt.plot(Xa, A['DPPM'], 'b--', label = f'Actual (last {-l} months)')
    plt.plot(Xb, m, 'r-', lw = 2, label = '$\mu$')
    plt.plot(Xa, ya, 'ko', lw = 2, label = '$y_{A}$')
    plt.fill_between(Xb.ravel(), m - 2 * s, m + 2 * s, color = 'red', alpha = 0.2, label = '$2\sigma$')
    plt.legend()
    plt.savefig('./labs/10/03b.pdf')
    plt.show()

    #def k(x, y):
    #    sn = np.sin(np.pi * np.sqrt(np.dot(x, y.T)))
    #    season = 4 * rbf(x, y, 0.5) * (np.exp(-2.0 / (1.0 ** 2) * sn * sn))
    #    irregular = 0.25 / (1 + np.dot(x, y.T) / 2.0)
    #    noise = 0.01 * rbf(x, y, 0.1) + np.array([[0.01 if np.abs(yj - xi) < 0.001 else 0 for yj in y] for xi in x])
    #    return season + irregular + noise