import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf

df = pd.read_csv("multiTimeline.csv", skiprows=1)
df["Semana"] = pd.to_datetime(df["Semana"])

df = df.set_index("Semana")
y = df["Pokémon GO: (Mundo)"]

print("\n Informações da Série")
print("======================================================")

print(y.describe())

plt.figure(figsize=(14,5))
plt.plot(y,label="Interesse pelo PokeGO")
plt.title("Interesse pelo PokeGO ao longo do tempo")
plt.xlabel("Data")
plt.ylabel("Indice Google Trends")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14,5))
plt.plot(y, alpha=0.25, label="Original")
plt.plot(y.rolling(4).mean(), label="MM 4 semanas")
plt.plot(y.rolling(8).mean(), label="MM 8 semanas")
plt.plot(y.rolling(13).mean(), label="MM 13 semanas")

plt.title("Comparação medias moveis")
plt.xlabel("Data")
plt.ylabel("Interesse")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

##decomposiçao sazonal
decomposicao = seasonal_decompose(y, model="additive", period=52)

fig = decomposicao.plot()
fig.set_size_inches(14,9)
fig.suptitle("Decomposição sazonal pokego", y=1.02)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))

plot_acf(
    y,
    lags=104,
    zero=False
)

plt.title("Função de Autocorrelação - Pokémon GO")
plt.xlabel("Defasagem (semanas)")
plt.tight_layout()
plt.show()