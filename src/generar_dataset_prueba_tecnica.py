"""
Generador del dataset para la Prueba Tecnica — Analista / Especialista de Datos, Lending.

Este script simula una base de clientes de una campana de Prestamos Personales
(Adquisicion y Recompra/Restitucion), con la informacion sociodemografica,
de comportamiento crediticio y de resultado de campana descrita en el
diccionario de datos.

Uso:
    python generar_dataset_prueba_tecnica.py

Esto genera el archivo `dataset_prueba_tecnica.csv` en el directorio actual.
Requiere: pandas, numpy
"""

import numpy as np
import pandas as pd

SEED = 42
N_FILAS = 250000

rng = np.random.default_rng(SEED)
N = N_FILAS

# --- Variables sociodemograficas y de relacion con el banco -----------------
tipo_cliente = rng.choice(["Nuevo", "Recompra"], size=N, p=[0.6, 0.4])
edad = np.clip(rng.normal(38, 11, N), 21, 70).round().astype(int)

antiguedad_meses = np.where(
    tipo_cliente == "Nuevo",
    0,
    np.clip(rng.gamma(3.0, 22, N), 6, 240).round().astype(int),
)

region = rng.choice(
    ["CDMX", "Norte", "Occidente", "Bajio", "Sureste"],
    size=N, p=[0.28, 0.22, 0.18, 0.17, 0.15],
)
region_income_mult = pd.Series(region).map(
    {"CDMX": 1.35, "Norte": 1.15, "Occidente": 1.0, "Bajio": 0.95, "Sureste": 0.85}
).to_numpy()

ingreso_base = rng.lognormal(mean=9.9, sigma=0.45, size=N)
ingreso_mensual_mxn = np.clip(ingreso_base * region_income_mult, 6000, 180000).round(0)

z_ingreso = (np.log(ingreso_mensual_mxn) - np.log(ingreso_mensual_mxn).mean()) / np.log(ingreso_mensual_mxn).std()
z_antig = (antiguedad_meses - antiguedad_meses.mean()) / (antiguedad_meses.std() + 1e-6)
score_buro = np.clip(
    620 + 55 * z_ingreso + 25 * z_antig + rng.normal(0, 55, N), 300, 850
).round().astype(int)

num_productos_activos = np.where(
    tipo_cliente == "Nuevo", 0,
    np.clip(rng.poisson(1.6, N), 0, 6)
)

uso_tdc_pct = np.clip(rng.normal(55, 28, N) - 0.02 * (score_buro - 620), 0, 140).round(1)
num_consultas_buro_6m = rng.poisson(2.0, N)
tiene_mora_actual = np.clip(
    (rng.uniform(0, 1, N) < (0.14 - 0.0001 * (score_buro - 620))).astype(int), 0, 1
)

# --- Canal de contacto y variables de la oferta ------------------------------
canal_probs = []
for r in region:
    if r in ("CDMX", "Norte"):
        canal_probs.append([0.35, 0.30, 0.20, 0.15])  # App, Sucursal, CallCenter, Web
    else:
        canal_probs.append([0.25, 0.20, 0.30, 0.25])
canal_probs = np.array(canal_probs)
canales = ["App", "Sucursal", "Call Center", "Web"]
canal_preferente = np.array([rng.choice(canales, p=canal_probs[i]) for i in range(N)])

num_contactos_previos_12m = rng.poisson(2.3, N)
dias_desde_ultimo_contacto = rng.integers(1, 365, N)

oferta_tasa_pct = np.clip(rng.normal(26, 5, N) - 0.01 * (score_buro - 620), 14, 42).round(2)
monto_ofertado_mxn = np.clip(
    (ingreso_mensual_mxn * rng.uniform(2.0, 5.5, N)) * (0.7 + 0.3 * (score_buro / 850)),
    15000, 300000
).round(-2)

# --- Grupo experimental de la campana ----------------------------------------
grupo = rng.choice(["Tratamiento", "Control"], size=N, p=[0.85, 0.15])
contactado = (grupo == "Tratamiento").astype(int)
costo_contacto_mxn = np.select(
    [canal_preferente == "App", canal_preferente == "Web",
     canal_preferente == "Call Center", canal_preferente == "Sucursal"],
    [12, 18, 75, 150],
) * contactado

# --- Probabilidad de contratacion --------------------------------------------
z_score = (score_buro - 620) / 100
z_tasa = -(oferta_tasa_pct - 26) / 5
canal_ajuste = {"App": 0.10, "Web": 0.05, "Call Center": 0.0, "Sucursal": 0.02}
canal_effect = pd.Series(canal_preferente).map(canal_ajuste).to_numpy()
tipo_effect = np.where(tipo_cliente == "Recompra", 0.25, 0.0)
mora_penalty = -0.35 * tiene_mora_actual
uso_tdc_effect = 0.15 * (uso_tdc_pct / 100)
efecto_contacto = 0.55

logit = (
    -2.6
    + 0.65 * z_score
    + 0.35 * z_ingreso
    + 0.55 * z_tasa
    + canal_effect
    + tipo_effect
    + mora_penalty
    + uso_tdc_effect
    - 0.05 * (num_contactos_previos_12m > 5)
    + efecto_contacto * contactado
    + rng.normal(0, 0.4, N)
)
prob_contrato = 1 / (1 + np.exp(-logit))
contrato = (rng.uniform(0, 1, N) < prob_contrato).astype(int)

monto_contratado_mxn = np.where(
    contrato == 1,
    np.clip(monto_ofertado_mxn * rng.uniform(0.6, 1.0, N), 10000, 300000).round(-2),
    0,
)

# --- Riesgo posterior a la originacion ---------------------------------------
z_score_risk = -(score_buro - 620) / 100
logit_mora = -2.2 + 0.9 * z_score_risk + 0.9 * (uso_tdc_pct / 100) + rng.normal(0, 0.5, N)
prob_mora = 1 / (1 + np.exp(-logit_mora))
mora_90d_post_6m = np.where(contrato == 1, (rng.uniform(0, 1, N) < prob_mora).astype(int), np.nan)

df = pd.DataFrame({
    "id_cliente": [f"C{100000+i}" for i in range(N)],
    "tipo_cliente": tipo_cliente,
    "region": region,
    "edad": edad,
    "antiguedad_meses": antiguedad_meses,
    "ingreso_mensual_mxn": ingreso_mensual_mxn,
    "score_buro": score_buro,
    "num_productos_activos": num_productos_activos,
    "uso_tdc_pct": uso_tdc_pct,
    "num_consultas_buro_6m": num_consultas_buro_6m,
    "tiene_mora_actual": tiene_mora_actual,
    "canal_preferente": canal_preferente,
    "num_contactos_previos_12m": num_contactos_previos_12m,
    "dias_desde_ultimo_contacto": dias_desde_ultimo_contacto,
    "oferta_tasa_pct": oferta_tasa_pct,
    "monto_ofertado_mxn": monto_ofertado_mxn,
    "grupo_experimental": grupo,
    "contactado": contactado,
    "costo_contacto_mxn": costo_contacto_mxn,
    "contrato": contrato,
    "monto_contratado_mxn": monto_contratado_mxn,
    "mora_90d_post_6m": mora_90d_post_6m,
})

# --- Incidencias de captura tipicas de sistemas operacionales ----------------
n_incidencias = max(8, int(N * 0.0006))
idx_incidencias = rng.choice(N, size=n_incidencias, replace=False)
mitad = n_incidencias // 2
for j, i in enumerate(idx_incidencias[:mitad]):
    tipo_error = j % 3
    if tipo_error == 0:
        df.loc[i, "edad"] = int(rng.choice([3, 130, 145]))
    elif tipo_error == 1:
        df.loc[i, "ingreso_mensual_mxn"] = -abs(df.loc[i, "ingreso_mensual_mxn"])
    else:
        df.loc[i, "score_buro"] = int(rng.choice([120, 999, 1500]))
donantes = rng.choice(N, size=n_incidencias - mitad, replace=False)
for i, donante in zip(idx_incidencias[mitad:], donantes):
    df.loc[i, "id_cliente"] = df.loc[donante, "id_cliente"]

# --- Guardar dataset ----------------------------------------------------------
OUTPUT_PATH = "dataset_prueba_tecnica.csv"
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"Dataset generado: {OUTPUT_PATH}")
print(f"Filas: {len(df):,} | Columnas: {df.shape[1]}")
