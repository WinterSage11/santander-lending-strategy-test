# Estrategia de Priorización - Campaña Lending (Santander)

## Contexto del Proyecto
Este repositorio contiene la resolución técnica para la priorización de la próxima campaña de Préstamos Personales. El análisis busca optimizar el 25% de la capacidad de contacto disponible mediante un enfoque de inferencia causal y rentabilidad ajustada por riesgo.

## Estructura del Repositorio
* `src/`: Contiene el script generador de la base de datos sintética.
* `data/`: Carpeta destino para el dataset generado localmente (ignorado en Git por seguridad).
* `notebooks/`: Contiene el Jupyter Notebook principal (`01_analisis_y_estrategia_lending.ipynb`) con la resolución paso a paso de las Preguntas 1 a 7.
* `presentacion/`: Contiene la síntesis ejecutiva de 5 slides (Pregunta 8) orientada al Comité de Negocio.

## Instrucciones de Ejecución
1. Instalar las dependencias listadas en `requirements.txt`.
2. Ejecutar `python src/generar_dataset_prueba_tecnica.py` para poblar la carpeta `data/` con el archivo CSV.
3. Ejecutar el notebook en `notebooks/` de principio a fin. El código está estructurado de manera secuencial.

## Hallazgos Principales
* Se implementó un modelo T-Learner para aislar el Uplift real de la campaña.
* Se desarrolló un score de Rentabilidad Ajustada por Riesgo, desplazando a 12,086 clientes de alta propensión pero con alto riesgo de default.
* Se propone una automatización del monitoreo de KPIs operativos y de riesgo utilizando orquestación y modelos fundacionales (LLMs).