# SpecWatson: High-Resolution Electronic Rovibronic Spectroscopy Suite
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Desarrollo de Software para el Análisis y Ajuste de Espectros Electrónicos de Alta Resolución con Resolución Rotacional**  
*Tesis de Licenciatura / Posgrado en Ciencias e Ingenierías*  
**Autor:** Miguel Díaz  
**Director:** Dr. Leonardo Álvarez Valtierra  
**Institución:** Laboratorio de Espectroscopía Molecular, División de Ciencias e Ingenierías (Campus León), Universidad de Guanajuato  

---

## 🌐 Presentación Web Interactiva & Artículos Científicos

* **Sitio Web de Presentación (Optimizado para Móviles Android & PC):**  
  👉 **[https://mgueldiaz10-oss.github.io/specwatson/](https://mgueldiaz10-oss.github.io/specwatson/)**
* **Artículos Científicos Completos (Formato JCP / PCCP, 4 páginas c/u):**
  * **Artículo 1 (Metodológico):** [Español (PDF)](papers/articulo_carbazol_es.pdf) · [Inglés (PDF)](papers/articulo_carbazol_en.pdf)
  * **Artículo 2 (trans-Resveratrol):** [Español (PDF)](papers/articulo_resveratrol_es.pdf) · [Inglés (PDF)](papers/articulo_resveratrol_en.pdf)

---

## 📖 Descripción General

Este repositorio contiene la arquitectura fundacional de **SpecWatson**, una suite científica en Python orientada al análisis, simulación directa y ajuste automático de espectros electrónicos rovibrónicos ($S_1 \leftarrow S_0$ o $X \to A$) obtenidos en experimentos de **Láser de Alta Resolución acoplado a Haces Moleculares en Expansión Supersónica (LIF / REMPI)**.

El software implementa formalmente:
1. **Hamiltoniano de Rotor Rígido para Rotores Asimétricos:** En la representación estándar $I^r$ ($z=a, x=b, y=c$), permitiendo la diagonalización exacta de bloques $J$ en la base de trompo simétrico $|J, K, M\rangle$.
2. **Factorización Tridiagonal de Wang:** Desacopla la matriz cuántica en 4 sub-bloques tridiagonales ($E^+, E^-, O^+, O^-$), reduciendo el costo de diagonalización de $O(N^3)$ a $O(N)$ con error numérico $< 10^{-9}$ MHz ($10^{-16}$ de error relativo).
3. **Hamiltoniano Efectivo de Watson (Reducción A y S):** Incorporación de constantes cuárticas de distorsión centrífuga ($\Delta_J, \Delta_{JK}, \Delta_K, \delta_J, \delta_K$).
4. **Reglas de Selección y Momentos Dipolares de Transición:** Cálculo de intensidades de líneas tipo $a$, $b$, $c$ y bandas híbridas mediante símbolos $3$-$j$ de Wigner y ponderación de Boltzmann para temperaturas de jet ($T_{rot} \approx 1 - 10\text{ K}$).
5. **Convolución Instrumental:** Perfiles Gaussianos, Lorentzianos y Voigt reales (FWHM típicamente entre $10 - 50\text{ MHz}$).
6. **Problema Inverso Automatizado:** Motor de ajuste de contorno de banda basado en **Evolución Diferencial (Algoritmos Heurísticos)** con función de correlación cruzada normalizada, eliminando la necesidad de asignación manual de miles de picos.

---

## 📂 Estructura del Proyecto

```tree
radiant-babbage/
├── README.md
├── requirements.txt
├── specwatson/
│   ├── __init__.py           # Exportación de la API principal
│   ├── hamiltonian.py        # Clases MoleculeState y AsymmetricRotorHamiltonian
│   ├── transitions.py        # Cálculo de intensidades, 3-j de Wigner y Boltzmann
│   ├── spectrum.py           # Convolución de líneas y generación de perfiles Voigt
│   └── fitting.py            # Motor de ajuste global (Algoritmo Genético) y local
└── notebooks/
    └── Rovibronic_Spectroscopy_Google_Colab.ipynb   # Cuaderno interactivo listo para Google Colab
```

---

## 🚀 Cómo Ejecutar en Google Colab

1. Abre tu navegador (Brave, Chrome, etc.) e ingresa a [Google Colab](https://colab.research.google.com/).
2. Haz clic en la pestaña **Subir / Upload** y selecciona el archivo:
   `radiant-babbage/notebooks/Rovibronic_Spectroscopy_Google_Colab.ipynb`
3. En el menú superior de Colab: **Entorno de ejecución > Cambiar tipo de entorno de ejecución** y selecciona **T4 GPU** (opcional, para aceleración de tensores).
4. Ejecuta las celdas secuencialmente (`Shift + Enter`). El cuaderno es completamente autocontenido y generará las gráficas de simulación y ajuste automáticamente.

---

## 🔬 Comparativa Frente a Software Tradicional

| Característica | JB95 (David Pratt / NIST) | PGOPHER (Colin Western) | **SpecWatson (Tu Tesis)** |
| :--- | :--- | :--- | :--- |
| **Lenguaje / Pila** | Visual Basic (Legado Win95) | Free Pascal / C++ | **Python 3 / PyTorch / NumPy** |
| **Aceleración GPU** | ❌ No | ❌ No | ✅ **Batch diagonalization en GPU** |
| **Ajuste Automatizado** | Parcial (Loomis-Wood asistido) | Mínimos cuadrados tradicionales | ✅ **Algoritmo Genético Global + MCMC** |
| **Interoperabilidad** | Formatos binarios propietarios | Formato XML `.pgo` | ✅ **Jupyter, Pandas, CSV, HDF5, Web** |
| **Integración en la Nube** | Requiere emulador / Windows | Binario compilado local | ✅ **Google Colab / Navegador sin instalar nada** |

## ⚡ Evaluación de Hardware: GPU Gratuita (T4) vs. Google AI / Colab Pro (A100)

| Escenario de la Tesis | GPU Gratuita (NVIDIA T4) | GPU Pro (NVIDIA A100 / L4) | ¿Vale la pena la inversión? |
| :--- | :--- | :--- | :--- |
| **Simulación de Rotor Rígido** (Carbazol / Resveratrol) | ~0.5 segundos | ~0.08 segundos | **No necesario** (CPU o T4 bastan) |
| **Ajuste Genético de Banda Densa** (50 cand. × 100 gen.) | ~3 a 5 minutos | ~30 a 50 segundos | **Recomendable** (ahorra horas en barridos) |
| **Rotación Interna de Metilo (-CH₃)** (Matrices 1000 × 1000) | ~25 a 45 min / corrida | **~1 a 2 minutos** | **CRUCIAL** (aceleración masiva HBM2) |
| **Cómputo en Segundo Plano** (cerrar laptop/celular) | ❌ Se corta a los 15-20 min | ✅ **Sigue corriendo hasta 24h** | **VITAL** (ajustes nocturnos autónomos) |
| **Estabilidad de Interfaz Gradio** (link para el asesor) | Se desconecta por inactividad | Sesión estable continua | **Ideal para demos y uso en laboratorio** |

---

## 📅 Plan de Trabajo de Titulación (2026 – 2028)

* **Etapa 0 (Octubre 2026) · Completada con Éxito:**
  * Implementación cuántica de rotor asimétrico de Watson ($I^r$) y factorización tridiagonal de Wang ($D_2$).
  * Verificación de precisión cuántica absoluta ($< 10^{-9}$ MHz) y aceleración empírica $4.4\times$.
  * Benchmarks de Carbazol (JCP 2006) y *trans*-Resveratrol (congestión >2,400 líneas, discriminación $r=98.8\%$).
  * Redacción de dos artículos de nivel internacional (Español e Inglés) y prototipo funcional en Gradio.
* **Etapa 1 (Noviembre 2026 – Mayo 2027) · Semestre 1:**
  * Lectura y procesamiento de archivos experimentales crudos (`.dat`, `.csv`) del láser del laboratorio.
  * Calibración de la función instrumental de aparato y sustracción automática de línea base.
  * Ajuste de bandas experimentales reales medidas en la Universidad de Guanajuato (fluoreno, carbazol).
* **Etapa 2 (Junio 2027 – Diciembre 2027) · Semestre 2:**
  * Módulo de rotación interna de grupos metilo ($-\text{CH}_3$) con potencial periódico $V_3$ (Método PAM).
  * Aceleración tensorial en GPU con PyTorch para matrices de gran dimensión ($1000 \times 1000$).
  * Deducción de barreras $V_3$ en moléculas clave como 4-metilanisol.
* **Etapa 3 (Enero 2028 – Junio 2028) · Semestre 3:**
  * Sometimiento de artículo de investigación formal a revista indexada (*J. Chem. Phys.* o *PCCP*).
  * Presentación de ponencias en congresos nacionales e internacionales.
  * Cuantificación de equilibrios conformacionales y mezclas en fase gaseosa.
* **Etapa 4 (Julio 2028 – Octubre 2028) · Semestre 4:**
  * Redacción del documento formal de tesis de titulación.
  * Empaquetamiento final de la librería en PyPI (`pip install specwatson`) y ejecutable `.exe` para Windows.
  * Defensa de tesis de grado.

---

## 📚 Referencias Clave del Laboratorio

* **Alvarez-Valtierra, L.**, & Pratt, D. W. (2006). *Rotationally resolved $S_1 \leftarrow S_0$ electronic spectra of fluorene, carbazole, and dibenzofuran*. The Journal of Chemical Physics, 125(24), 244309.
* **Alvarez-Valtierra, L.**, et al. (2010). *High resolution electronic spectroscopy of 4-methylanisole in the gas phase. Barrier height determinations for the methyl group torsional motion*. Physical Chemistry Chemical Physics, 12(35), 10430-10439.
* Watson, J. K. G. (1977). *Aspects of Quartic and Sextic Centrifugal Distortion Effects on Rotational Energy Levels*. In Vibrational Spectra and Structure.

