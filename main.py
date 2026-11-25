import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox


df = pd.read_csv("multiTimeline.csv", skiprows=1)
df["Semana"] = pd.to_datetime(df["Semana"])

df = df.set_index("Semana")
y = df["decoração de Natal: (Brasil)"].astype(str).str.strip()

y = y.replace("<1", "0.5")
y = pd.to_numeric(y)

print("\nVALIDAÇÃO DA SÉRIE TEMPORAL")
print("=" * 50)

print("Primeira observação:", y.index.min())
print("Última observação:", y.index.max())
print("Total de observações:", len(y))
print("Frequência identificada:", pd.infer_freq(y.index))
print("Datas duplicadas:", y.index.duplicated().sum())
print("Valores ausentes:", y.isna().sum())
print("Índice ordenado:", y.index.is_monotonic_increasing)

print("\nESTATÍSTICAS DESCRITIVAS")
print(y.describe())


plt.figure(figsize=(14,5))
plt.plot(y,label="Interesse por decoração de Natal")
plt.title("Interesse por decoração de Natal ao longo do tempo")
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
fig.suptitle("Decomposição sazonal Deco Natal", y=1.02)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))

plot_acf(
    y,
    lags=104,
    zero=False
)

plt.title("Função de Autocorrelação - Decoração Natal")
plt.xlabel("Defasagem (semanas)")
plt.tight_layout()
plt.show()

y_sazonal = y.diff(52)

plt.figure(figsize=(14, 5))
plt.plot(y_sazonal)

plt.title("Série após diferenciação sazonal (52 semanas)")
plt.xlabel("Data")
plt.ylabel("Diferença sazonal")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ACF após diferenciação sazonal
plt.figure(figsize=(12, 5))

plot_acf(
    y_sazonal.dropna(),
    lags=104,
    zero=False
)

plt.title("ACF após diferenciação sazonal")
plt.xlabel("Defasagem (semanas)")
plt.tight_layout()
plt.show()

def testar_estacionariedade(serie, nome):
    serie = serie.dropna()

    print(f"\n{'=' * 50}")
    print(f"TESTES DE ESTACIONARIEDADE: {nome}")
    print("=" * 50)

    # ADF
    resultado_adf = adfuller(serie)

    print("\nADF:")
    print(f"Estatística: {resultado_adf[0]:.4f}")
    print(f"p-valor: {resultado_adf[1]:.4f}")

    # KPSS
    resultado_kpss = kpss(serie, regression="c", nlags="auto")

    print("\nKPSS:")
    print(f"Estatística: {resultado_kpss[0]:.4f}")
    print(f"p-valor: {resultado_kpss[1]:.4f}")


testar_estacionariedade(y, "Série original")
testar_estacionariedade(y_sazonal, "Série com diferenciação sazonal")

# PACF da série original
fig, ax = plt.subplots(figsize=(12, 5))

plot_pacf(
    y,
    lags=52,
    method="ywm",
    zero=False,
    ax=ax
)

ax.set_title("PACF - Série original")
ax.set_xlabel("Defasagem (semanas)")
plt.tight_layout()
plt.show()


# PACF após diferenciação sazonal
fig, ax = plt.subplots(figsize=(12, 5))

plot_pacf(
    y_sazonal.dropna(),
    lags=52,
    method="ywm",
    zero=False,
    ax=ax
)

ax.set_title("PACF - Série com diferenciação sazonal")
ax.set_xlabel("Defasagem (semanas)")
plt.tight_layout()
plt.show()

# Modelos candidatos
# Melhor modelo até agora
modelo = SARIMAX(
    y,
    order=(0, 0, 1),
    seasonal_order=(0, 1, 1, 52),
    trend="n",
    enforce_stationarity=False,
    enforce_invertibility=False
)

ajuste = modelo.fit(disp=False)

print(ajuste.summary())

# Resíduos
residuos = ajuste.resid.iloc[52:]

# Resíduos do modelo
# Desconsiderar as primeiras 52 observações devido à inicialização sazonal
residuos = ajuste.resid.iloc[52:]

# Gráfico dos resíduos
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(residuos)
axes[0].set_title("Resíduos do SARIMA(0,0,1)(0,1,1,52)")
axes[0].axhline(0, color="black", linestyle="--")
axes[0].grid(alpha=0.3)

plot_acf(
    residuos,
    lags=104,
    zero=False,
    ax=axes[1]
)

axes[1].set_title("ACF dos resíduos")
plt.tight_layout()
plt.show()

# Teste de Ljung-Box
teste = acorr_ljungbox(
    residuos,
    lags=[13, 26, 52],
    model_df=2,
    return_df=True
)

print("\nDIAGNÓSTICO LJUNG-BOX")
print(teste)



# Comparação entre D=0 e D=1

modelos_comparacao = {
    "D=0": (0, 0, 1, 52),
    "D=1": (0, 1, 1, 52)
}

for nome, ordem_sazonal in modelos_comparacao.items():

    modelo_comp = SARIMAX(
        y,
        order=(0, 0, 1),
        seasonal_order=ordem_sazonal,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    ajuste_comp = modelo_comp.fit(disp=False)

    residuos_comp = ajuste_comp.resid.iloc[52:]

    ljung = acorr_ljungbox(
        residuos_comp,
        lags=[13, 26, 52],
        model_df=2,
        return_df=True
    )

    print(f"\nMODELO {nome}")
    print("AIC:", ajuste_comp.aic)
    print("BIC:", ajuste_comp.bic)
    print("\nLjung-Box:")
    print(ljung)

# PREVISÃO PARA AS PRÓXIMAS 52 SEMANAS

previsao = ajuste.get_forecast(steps=52)

media = previsao.predicted_mean
intervalo = previsao.conf_int()

plt.figure(figsize=(14, 6))

# Série histórica
plt.plot(y, label="Valores observados")

# Previsão
plt.plot(media, color="red", label="Previsão")

# Intervalo de confiança
plt.fill_between(
    media.index,
    intervalo.iloc[:, 0],
    intervalo.iloc[:, 1],
    color="red",
    alpha=0.15,
    label="Intervalo de confiança (95%)"
)

plt.title("Previsão do interesse por decoração de Natal")
plt.xlabel("Data")
plt.ylabel("Índice Google Trends")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\nPREVISÃO DAS PRÓXIMAS 52 SEMANAS")
print(media)


# VALIDAÇÃO FORA DA AMOSTRA (OUT-OF-SAMPLE)

# Separar os dados: últimas 52 semanas para teste
y_treino = y.iloc[:-52]
y_teste = y.iloc[-52:]

print("\nVALIDAÇÃO DO MODELO")
print("Observações para treino:", len(y_treino))
print("Observações para teste:", len(y_teste))
print("Início do teste:", y_teste.index.min())
print("Fim do teste:", y_teste.index.max())

# Ajustar o SARIMA somente com os dados de treino
modelo_teste = SARIMAX(
    y_treino,
    order=(0, 0, 1),
    seasonal_order=(0, 1, 1, 52),
    trend="n",
    enforce_stationarity=False,
    enforce_invertibility=False
)

ajuste_teste = modelo_teste.fit(disp=False)

# Prever as 52 semanas reservadas
previsao_teste = ajuste_teste.get_forecast(steps=52)
valores_previstos = previsao_teste.predicted_mean

# Comparar valores previstos com valores reais
comparacao = pd.DataFrame({
    "Real": y_teste,
    "Previsto": valores_previstos
})

# Métricas de erro
mae = np.mean(np.abs(comparacao["Real"] - comparacao["Previsto"]))
rmse = np.sqrt(np.mean((comparacao["Real"] - comparacao["Previsto"]) ** 2))

print("\nMÉTRICAS DE AVALIAÇÃO")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# Modelo de referência: previsão sazonal ingênua
# Repete os valores observados no ano anterior
previsao_ingenua = pd.Series(
    y_treino.iloc[-52:].values,
    index=y_teste.index
)

mae_ingenuo = np.mean(np.abs(y_teste - previsao_ingenua))
rmse_ingenuo = np.sqrt(np.mean((y_teste - previsao_ingenua) ** 2))

print("\nMODELO DE REFERÊNCIA (SAZONAL INGÊNUO)")
print(f"MAE: {mae_ingenuo:.4f}")
print(f"RMSE: {rmse_ingenuo:.4f}")

# Gráfico da comparação
plt.figure(figsize=(14, 6))

plt.plot(y_treino, label="Treino", alpha=0.5)
plt.plot(y_teste, label="Valores reais (teste)", color="blue")
plt.plot(valores_previstos, label="Previsão SARIMA", color="red")
plt.plot(previsao_ingenua, label="Previsão sazonal ingênua",
         color="green", linestyle="--")

plt.title("Validação da previsão - últimas 52 semanas")
plt.xlabel("Data")
plt.ylabel("Índice Google Trends")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
