from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import polars as pl
import pandas as pd
import numpy as np
import scipy as sp

co2 = pd.read_csv('./labs/10/co2_daily_mlo.csv', comment='#', header=None,
                       names=["Year", "Month", "Day", "DecimalDate", "ppm"])

co2_data = pl.from_pandas(co2[['Year', 'Month', 'Day', 'ppm']]).select(
    pl.date("Year", "Month", "Day"), "ppm"
)

co2_data = (
    co2_data.sort(by="date")
    .group_by_dynamic("date", every="1mo")
    .agg(pl.col("ppm").mean())
    .drop_nulls()
)

def detrend_linear(X, y):
    model = LinearRegression()
    model.fit(X, y)

    trend = model.predict(X)
    return y - trend

def detrend_filter(yo):
    y = np.zeros_like(yo)
    for i in range(1, y.shape[0]):
        y[i] = yo[i] - yo[i - 1]

    return y

X = co2_data.select(
    pl.col("date").dt.year() + pl.col("date").dt.month() / 12
).to_numpy()

# y = detrend_linear(X, co2_data["ppm"].to_numpy())
y = detrend_filter(co2_data["ppm"].to_numpy())

def kper(x, y, alpha = 1.0, beta = 1.0, gamma = 1.0):
    dist = np.abs(x - y.T)
    sine_squared = np.sin(np.pi * dist / beta) ** 2
    return alpha * np.exp(-2 * sine_squared / (gamma ** 2))

def knoi(x, y, sigma = 1.0):
    if x.shape != y.shape:
        return 0.0
    
    kron = np.linalg.norm(x - y)
    kron = 1.0 if kron < 0.0001 else 0.0
    return sigma * sigma * kron

def klin(x, y, sigma = 0.05):
    return sigma * sigma * np.dot(x, y.T)

def krbf(x, y, alpha = 0.5):
    dist_sq = np.sum((x[:, None] - y[None, :])**2, axis=-1)
    return alpha * np.exp(-1.5 * dist_sq / (13.0 ** 2))

def k(x, y):
    # return kper(x, y) + knoi(x, y)
    return kper(x, y, 0.25, 1, 13.0) + krbf(x, y, 0.1)
    # return kper(x, y)

l = 120
A = X[-l:].reshape(-1, 1)
B = X[:-l].reshape(-1, 1)
ya = y[-l:]
yb = y[:-l]
ma = ya.mean()
mb = yb.mean()

CAA = k(A, A)
CAB = k(A, B)
CBA = k(B, A)
CBB = k(B, B)
CBB += 1e-4 * np.eye(len(CBB))
CBB_inv = sp.linalg.solve(CBB, np.eye(len(CBB)), assume_a = 'pos')

def covm(c):
    print(np.allclose(c, c.T))
    print(np.min(c))
    eigenvalues = np.linalg.eigvals(c)
    print(np.min(eigenvalues))

m = CAB @ CBB_inv @ (yb - mb)
D = CAA - (CAB @ CBB_inv @ CBA)
s = np.sqrt(np.diag(D))

plt.figure(figsize = (10, 6))
plt.plot(np.concatenate([B, A[:2]]), np.concatenate([yb, ya[:2]]), color="black", linestyle="dashed", label="Measurements")
plt.plot(A, m, color="orange", alpha=0.75, label="Gaussian process")
plt.fill_between(
    A.ravel(),
    m - s,
    m + s,
    color="tab:blue",
    alpha=0.2,
    label = 'Confidence ($±\sigma$)'
)
plt.legend()
plt.xlabel("Date")
plt.ylabel("Detrended CO$_2$ concentration (ppm)")
plt.tight_layout()
plt.savefig('./labs/10/03c1.pdf')
plt.show()

samples = np.random.multivariate_normal(m.flatten(), D, size = 25)
plt.figure(figsize = (10, 6))
plt.plot(A, ya, color = 'green', linestyle='dashed', label='Actual')
plt.plot(A, samples.T, color="tab:blue", alpha = 0.35)
plt.plot(A, samples[0], color="tab:blue", alpha = 0.35, label = 'Realisations')
plt.plot(A, m, color="black", alpha=0.95, label="Gaussian process", linewidth = 3.0)
plt.legend()
plt.xlabel("Date")
plt.ylabel("Detrended CO$_2$ concentration (ppm)")
plt.tight_layout()
plt.savefig('./labs/10/03c2.pdf')
plt.show()