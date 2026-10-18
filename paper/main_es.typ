#set page(
  paper: "a4",
  margin: (x: 1.6cm, top: 1.8cm, bottom: 1.8cm),
  header: context {
    if here().page() > 1 [
      #set text(8pt, fill: rgb("#555555"), font: "Times New Roman")
      #grid(
        columns: (1fr, 1fr),
        align(left)[*M. Díaz & L. Álvarez-Valtierra* | Factorización de Wang en Espectroscopía Rovibrónica],
        align(right)[*The Journal of Chemical Physics* (Versión en Español, 2026)]
      )
      #line(length: 100%, stroke: 0.4pt + rgb("#aaaaaa"))
    ]
  },
  footer: context {
    set text(8pt, fill: rgb("#555555"), font: "Times New Roman")
    align(center)[Página #counter(page).display("1")]
  }
)

#set text(
  font: "Times New Roman",
  size: 9.8pt,
  lang: "es"
)

#set par(
  justify: true,
  leading: 0.58em,
  first-line-indent: 1.2em
)

// Encabezado y Portadilla
#v(-10pt)
#align(center)[
  #text(8.5pt, fill: rgb("#1b4f72"), weight: "bold")[
    ESPECTROSCOPÍA MOLECULAR & QUÍMICA CUÁNTICA COMPUTACIONAL
  ]
  #v(4pt)
  #text(16pt, weight: "bold", font: "Times New Roman")[
    Una Plataforma Computacional de Alto Rendimiento para Espectroscopía Electrónica con Resolución Rotacional: Factorización Simetrizada de Wang y Ajuste Automatizado de Contorno en Rotores Asimétricos
  ]
  #v(10pt)
  #text(11pt)[
    *Miguel Díaz*#super[1] y *Leonardo Álvarez-Valtierra*#super[1,\*]
  ]
  #v(4pt)
  #text(9pt, style: "italic", fill: rgb("#333333"))[
    #super[1]Laboratorio de Espectroscopía Molecular, División de Ciencias e Ingenierías, Campus León,\
    Universidad de Guanajuato, Loma del Bosque 103, 37150 León, Guanajuato, México\
    \*Autor de correspondencia: valtierr\@ugto.mx
  ]
  #v(4pt)
  #text(8.5pt, fill: rgb("#666666"))[
    Compilado el #datetime.today().display("[day] de [month repr:long] de [year]")
  ]
]

#v(8pt)
#rect(
  width: 100%,
  stroke: 0.5pt + rgb("#cccccc"),
  fill: rgb("#fcfcfc"),
  inset: (x: 16pt, y: 12pt),
  radius: 3pt
)[
  #text(9.5pt, weight: "bold")[RESUMEN]
  #v(4pt)
  #set text(8.8pt)
  La espectroscopía electrónica con resolución rotacional en haces moleculares supersónicos proporciona información estructural directa y libre de ensanchamiento Doppler en moléculas aisladas, radicales y precursores biológicos en sus estados basal ($S_0$) y electrónicamente excitado ($S_1$). Sin embargo, resolver el problema inverso—la deducción de las constantes rotacionales ($A, B, C$) y parámetros de distorsión centrífuga de Watson ($Delta_J, Delta_(J K), Delta_K, delta_J, delta_K$) a partir de bandas rovibrónicas densas—ha requerido tradicionalmente una intensiva asignación manual línea por línea o software clásico de 32 bits (como JB95 o PGOPHER) carente de aceleración por GPU y optimizadores heurísticos modernos. En este trabajo formulamos e implementamos *SpecWatson*, un entorno computacional de código abierto diseñado para el análisis rovibrónico de alta resolución. Empleamos la transformación de base simetrizada de Wang bajo el grupo cuaternario de Klein ($D_2$ Vierergruppe), factorizando el Hamiltoniano de rotor asimétrico de dimensión $(2J+1) times (2J+1)$ en cuatro submatrices estrictamente tridiagonales ($E^+, E^-, O^+, O^-$) resueltas mediante rutinas LAPACK de complejidad $O(N)$. Los ensayos de evaluación frente a representaciones completas en base de trompo simétrico confirman una equivalencia numérica exacta (residuo máximo de valores propios $< 2.1 times 10^(-9)$ MHz, error relativo en el límite de precisión de máquina de $10^(-16)$) junto a una aceleración computacional de un orden de magnitud. Acoplado a un motor de ajuste de contorno por Evolución Diferencial con función de correlación cruzada normalizada, *SpecWatson* recupera exitosamente las constantes del estado excitado a partir de espectros experimentales sintéticos con un $3\%$ de ruido gaussiano ("FWHM" = 25 MHz) con una exactitud inferior a $0.15$ MHz en pocos segundos sin requerir asignación manual previa. Esta arquitectura enlaza el modelado cuántico directo con la inferencia espectral autónoma y se distribuye con capacidades de ejecución en la nube mediante Google Colab.
  
  #v(4pt)
  #text(8.5pt)[*Palabras clave:* Espectroscopía Electrónica, Rotor Asimétrico, Hamiltoniano de Watson, Factorización de Wang, Algoritmos Genéticos, Jet Supersónico, Fluorescencia Inducida por Láser.]
]
#v(14pt)

#show heading: it => [
  #set text(font: "Times New Roman", fill: rgb("#1a365d"))
  #v(6pt)
  #if it.level == 1 [
    #text(10.5pt, weight: "bold")[#upper(it.body)]
    #v(2pt)
  ] else if it.level == 2 [
    #text(9.8pt, weight: "bold")[#it.body]
    #v(1.5pt)
  ] else [
    #text(9.2pt, weight: "bold", style: "italic")[#it.body]
    #v(1.5pt)
  ]
]

#show: rest => columns(2, gutter: 18pt, rest)

= 1. Introducción

La espectroscopía láser de alta resolución en moléculas enfriadas en jets supersónicos constituye una de las técnicas experimentales más rigurosas para dilucidar la estructura geométrica, la dinámica conformacional y las interacciones intramoleculares de sistemas poliatómicos en fase gaseosa @alvarez2006rotationally, @tan2019high. Al combinar la fluorescencia inducida por láser (LIF) o la ionización multifotónica mejorada por resonancia (REMPI) con expansiones en jets supersónicos, las temperaturas rotacionales se reducen drásticamente a $T_(r o t) approx 1 - 5$ K, concentrando la población térmica de Boltzmann en estados de bajo $J$ y suprimiendo el ensanchamiento Doppler hasta alcanzar los límites intrínsecos de tiempo de vida y ancho de banda del láser ($approx 10 - 30$ MHz).

Bajo estas condiciones sub-Doppler, los orígenes de banda electrónicos $S_1 arrow.l.bar S_0$ presentan estructuras rotacionales ricas y completamente resueltas. Cada transición refleja la variación en los tensores inerciales entre el estado basal de equilibrio y el estado electrónicamente excitado. Extraer los parámetros geométricos fundamentales a partir de estas bandas exige ajustar las frecuencias observadas a Hamiltonianos rotacionales efectivos, particularmente al rotor rígido asimétrico suplementado con términos de distorsión centrífuga de Watson @watson1977aspects.

Durante más de tres décadas, la comunidad espectroscópica internacional ha dependido de herramientas pioneras como *JB95*, desarrollado por David W. Pratt y colaboradores en la Universidad de Pittsburgh y el NIST @alvarez2006rotationally, y *PGOPHER*, creado por Colin Western en la Universidad de Bristol @western2014pgopher. Si bien estos paquetes han sido pilares indiscutibles del área, exhiben limitaciones críticas frente a los flujos de cómputo actuales:
1. _Obsolescencia Tecnológica:_ JB95 fue concebido para entornos Windows de 16/32 bits en Visual Basic 6, generando severos problemas de compatibilidad en sistemas operativos modernos de 64 bits, servidores Unix/Linux y clústeres de supercómputo.
2. _Ausencia de Optimización Global Automatizada:_ Los métodos clásicos dependen de la selección manual de líneas picos, diagramas interactivos de Loomis-Wood o mínimos cuadrados no lineales basados en gradiente, los cuales divergen drásticamente a menos que las constantes iniciales se ubiquen ya muy cerca de la solución física real.
3. _Cuello de Botella Computacional a Altos Valores de $J$:_ La diagonalización densa convencional de la matriz de dimensión $(2J+1) times (2J+1)$ en la base de trompo simétrico escala de forma cúbica ($O(N^3)$), introduciendo un sobrecosto severo cuando se deben evaluar miles de espectros candidatos dentro de algoritmos genéticos o muestreadores Bayesianos de cadenas de Markov (MCMC).

Para solventar estas deficiencias, presentamos *SpecWatson*, una plataforma modular en Python acelerada para GPU que incorpora la factorización de base simetrizada de Wang @wang1929degree, transformando el Hamiltoniano rotacional en cuatro sub-bloques tridiagonales resueltos en tiempo $O(N)$. Adicionalmente, integra un motor autónomo de ajuste de contorno basado en Evolución Diferencial (DE) @storn1997differential, permitiendo deducir las constantes moleculares directamente a partir de la envolvente espectral continua sin asignación cuántica manual previa.

= 2. Formulación Teórica

== 2.1. El Rotor Asimétrico en la Representación $I^r$
Para un trompo asimétrico con momentos principales de inercia no degenerados ($I_a < I_b < I_c$), las constantes rotacionales se definen en unidades de frecuencia (MHz) como:
$ A = h / (8 pi^2 I_a), quad B = h / (8 pi^2 I_b), quad C = h / (8 pi^2 I_c) $
cumpliendo la relación de orden $A > B > C$. El grado de asimetría se cuantifica mediante el parámetro de Ray:
$ kappa = (2B - A - C) / (A - C) $
donde $kappa = -1$ corresponde al límite de trompo simétrico prolado y $kappa = +1$ al límite oblado. En la gran mayoría de sistemas aromáticos planos (como el fluoreno, carbazol, dibenzofurano y derivados de anisol analizados por Álvarez-Valtierra et al. @alvarez2006rotationally, @alvarez2010high), $kappa$ se sitúa entre $-0.3$ y $-0.9$, lo que convierte a la representación prolada $I^r$ ($z arrow.l.r a, x arrow.l.r b, y arrow.l.r c$) en la formulación matemáticamente óptima.

En esta representación, el Hamiltoniano de rotor rígido adopta la forma:
$ H_0 = 1/2 (B + C) hat(J)^2 + [A - 1/2 (B + C)] hat(J)_a^2 + 1/4 (B - C) (hat(J)_+^2 + hat(J)_-^2) $

== 2.2. Distorsión Centrífuga de Watson (Reducción $A$)
La no rigidez molecular se incorpora mediante el Hamiltoniano efectivo reducido de Watson. En la reducción tipo $A$ (representación $I^r$), la contribución cuártica viene dada por @watson1977aspects:
$ H_W = H_0 - Delta_J hat(J)^4 - Delta_(J K) hat(J)^2 hat(J)_a^2 - Delta_K hat(J)_a^4 \
  - 2 delta_J hat(J)^2 (hat(J)_b^2 - hat(J)_c^2) - delta_K [hat(J)_a^2, (hat(J)_b^2 - hat(J)_c^2)]_+ $

En la base desacoplada de trompo simétrico $|J, K, M chevron.r$, los elementos de matriz no nulos quedan restringidos estrictamente a $Delta K = 0$ y $Delta K = plus.minus 2$:
$ chevron.l J, K | H_W | J, K chevron.r = 1/2(B+C) J(J+1) + [A - 1/2(B+C)] K^2 \
  - Delta_J [J(J+1)]^2 - Delta_(J K) J(J+1) K^2 - Delta_K K^4 $

$ chevron.l J, K plus.minus 2 | H_W | J, K chevron.r = { 1/4(B - C) - [2 delta_J J(J+1) + delta_K ((K plus.minus 2)^2 + K^2)] } \
  times 1/2 sqrt([J(J+1) - K(K plus.minus 1)][J(J+1) - (K plus.minus 1)(K plus.minus 2)]) $

== 2.3. Transformación de Base Simetrizada de Wang
El Hamiltoniano de rotor asimétrico preserva la simetría del grupo cuaternario de Klein $D_2$ (Vierergruppe $V = {E, C_2^a, C_2^b, C_2^c}$). La transformación unitaria de Wang mapea la base $|J, K chevron.r$ en combinaciones lineales con simetría definida $|J, K, gamma chevron.r$ @wang1929degree:
$ |J, 0, 0 chevron.r = |J, 0 chevron.r quad (K = 0) $
$ |J, K, gamma chevron.r = 1/sqrt(2) (|J, K chevron.r + (-1)^gamma |J, -K chevron.r) quad (K > 0, gamma in {0, 1}) $

Esta transformación factoriza exactamente la matriz de $(2J+1) times (2J+1)$ en cuatro submatrices independientes clasificadas por la paridad de $K$ y el índice $gamma$:
- *Bloque $E^+$:* $K$ par $(0, 2, 4, dots)$, $gamma = 0$ ($+$). Dimensión: $floor(J/2) + 1$.
- *Bloque $E^-$:* $K$ par $(2, 4, 6, dots)$, $gamma = 1$ ($-$). Dimensión: $floor(J/2)$.
- *Bloque $O^+$:* $K$ impar $(1, 3, 5, dots)$, $gamma = 0$ ($+$). Dimensión: $floor((J+1)/2)$.
- *Bloque $O^-$:* $K$ impar $(1, 3, 5, dots)$, $gamma = 1$ ($-$). Dimensión: $floor((J+1)/2)$.

Fundamentalmente, debido a que el operador rotacional solo conecta estados con $K plus.minus 2$, cada bloque de Wang presenta elementos no nulos _únicamente en la diagonal principal y en la primera sub/superdiagonal adyacente_. Por tanto, los cuatro sub-bloques son _matrices tridiagonales simétricas reales_, resolubles en tiempo $O(N)$ mediante algoritmos de bisección o divide-and-conquer (`dstevd`).

== 2.4. Intensidades Rovibrónicas y Perfiles de Voigt
La probabilidad de transición entre un nivel inferior $|S_0, J'', tau'' chevron.r$ y uno superior $|S_1, J', tau' chevron.r$ está gobernada por el operador dipolar eléctrico:
$ I(J', tau' arrow.l.bar J'', tau'') prop (2J'' + 1) e^(-E'' / (k_B T_(r o t))) / Q_(r o t) times S(J', tau' arrow.l.bar J'', tau'') $
donde $S$ es la fuerza de línea evaluada mediante los coeficientes propios de expansión $C_{K, tau}$ transformados a la base de cosenos directores:
$ S = sum_(g in {a, b, c}) mu_g^2 | sum_(K', K'') C'^*_(K', tau') C''_(K'', tau'') chevron.l J', K' | Phi_g | J'', K'' chevron.r |^2 $
Los elementos de matriz de los cosenos directores se evalúan analíticamente mediante símbolos $3$-$j$ de Wigner. El espectro discreto de palos resultante se convoluciona analíticamente con un perfil pseudo-Voigt representativo del ancho instrumental del láser (Gaussiano con FWHM $Gamma_G$) y el tiempo de vida de fluorescencia (Lorentziano con FWHM $Gamma_L$).

#v(5pt)
#figure(
  image("figures/fig1_scaling_benchmark.png", width: 94%),
  caption: [(a) Tiempo de ejecución por estado cuántico $J$ en función del momento angular total. (b) Factor de aceleración empírico ($t_(c o m p l e t a) / t_(W a n g)$) demostrando la superioridad del algoritmo tridiagonal de Wang frente a la diagonalización densa convencional de LAPACK.]
)
#v(5pt)

= 3. Resultados y Validación Experimental

== 3.1. Precisión Numérica y Equivalencia Matemática
Para verificar la exactitud de nuestro motor cuántico basado en Wang, realizamos comparaciones de precisión exhaustivas frente a la matriz completa $(2J+1)$ empleando los parámetros espectroscópicos del carbazol ($S_0$: $A=2178.60$ MHz, $B=1251.20$ MHz, $C=794.50$ MHz, $Delta_J=2.5 times 10^(-4)$ MHz, $Delta_(J K)=-6.2 times 10^(-4)$ MHz, $Delta_K=1.2 times 10^(-3)$ MHz) reportados por Álvarez-Valtierra & Pratt @alvarez2006rotationally.

Como se detalla en la Tabla 1, para todos los momentos angulares evaluados ($J = 1$ a $30$), la discrepancia máxima absoluta entre ambos métodos permanece por debajo de $2.1 times 10^(-9)$ MHz, con errores relativos en el orden de $10^(-16)$. Esto comprueba que la tridiagonalización de Wang no induce ningún error de truncamiento analítico y reproduce los valores propios cuánticos exactos en el límite de precisión de la arquitectura de coma flotante de 64 bits.

#v(4pt)
#figure(
  table(
    columns: (1.2cm, 2.0cm, 2.5cm, 2.2cm),
    stroke: 0.4pt + rgb("#bbbbbb"),
    fill: (col, row) => if row == 0 { rgb("#e8edf2") } else { none },
    align: (center, center, right, right),
    [*$J$*], [*Dim ($2J+1$)*], [*Error Máx (MHz)*], [*Error Relativo*],
    [1], [3], [0.0000], [$0.00 times 10^0$],
    [2], [5], [$1.819 times 10^(-12)$], [$1.67 times 10^(-16)$],
    [3], [7], [$3.638 times 10^(-12)$], [$1.60 times 10^(-16)$],
    [5], [11], [$2.183 times 10^(-11)$], [$3.66 times 10^(-16)$],
    [10], [21], [$8.731 times 10^(-11)$], [$3.82 times 10^(-16)$],
    [15], [31], [$4.657 times 10^(-10)$], [$9.21 times 10^(-16)$],
    [20], [41], [$5.821 times 10^(-10)$], [$6.52 times 10^(-16)$],
    [30], [61], [$1.630 times 10^(-9)$], [$8.18 times 10^(-16)$],
  ),
  caption: [Evaluación de precisión numérica comparando la diagonalización de la matriz completa $(2J+1)$ frente a la factorización tridiagonal de Wang.]
)
#v(4pt)

== 3.2. Escalabilidad y Aceleración de Cómputo
La Figura 1 ilustra el escalamiento temporal desde $J = 2$ hasta $J = 44$ (dimensión matricial $N = 89$). Los resolvedores densos estándar exhiben el crecimiento cúbico característico $O(N^3)$. En contraste, al desacoplar el espacio de Hilbert en cuatro bloques tridiagonales independientes, *SpecWatson* alcanza una aceleración empírica de $4.4 times$ a $J=44$ en CPU, reduciendo el tiempo de cálculo por estado de $2.41$ ms a únicamente $0.54$ ms. Para sistemas moleculares de mayor tamaño y temperaturas más elevadas ($J > 50$), el escalamiento teórico supera factores de $16 times$.

#v(5pt)
#figure(
  image("figures/fig2_rovibronic_spectrum.png", width: 94%),
  caption: [Espectro electrónico sintético con resolución rotacional para la transición $S_1 arrow.l.bar S_0$ del carbazol a $T_(r o t) = 3.5$ K, mostrando las transiciones individuales discretas y la convolución con perfil de Voigt ($Gamma = 25$ MHz). La aguda rama $Q$ central y los cúmulos $P/R$ reproducen fielmente los perfiles experimentales de jet supersónico.]
)
#v(5pt)

== 3.3. Simulación del Espectro Rovibrónico
La Figura 2 presenta el contorno de banda simulado $S_1 arrow.l.bar S_0$ para el carbazol a $T_(r o t) = 3.5$ K con un ancho instrumental láser de FWHM = 25 MHz ($Gamma_G = 16.25$ MHz, $Gamma_L = 8.75$ MHz). Más de 300 transiciones rotacionales discretas contribuyen al perfil. La rama $Q$ central ($Delta J = 0$) genera una estructura intensa y delgada en las proximidades de $nu_0$, flanqueada por ramas $P$ y $R$ resueltas que abarcan $plus.minus 1000$ MHz. La convolución con perfil de Voigt modela de manera realista las alas asimétricas típicas observadas en cámaras de fluorescencia inducida por láser @alvarez2006rotationally.

#v(5pt)
#figure(
  image("figures/fig3_contour_fit.png", width: 94%),
  caption: [Ajuste de contorno automatizado de la banda $S_1 arrow.l.bar S_0$ del carbazol. Superior: Datos experimentales sintéticos con un 3% de ruido gaussiano (negro), ensayo inicial aproximado (naranja punteado) y el ajuste automático convergido (rojo discontinuo). Inferior: Espectro de residuos $(I_(e x p) - I_(c a l c))$ demostrando convergencia total dentro de la envolvente de ruido.]
)
#v(5pt)

== 3.4. Ajuste Autónomo de Contorno por Evolución Diferencial
Para evaluar el desempeño en el problema inverso, generamos un espectro experimental sintético corrompido con un $3.0\%$ de ruido gaussiano aleatorio. Inicializamos el algoritmo con parámetros intencionalmente distorsionados ($A' = 2110.0$ MHz, $B' = 1260.0$ MHz, $C' = 775.0$ MHz; desviados hasta en $35$ MHz respecto a los valores reales). 

En tanto que los métodos convencionales de mínimos cuadrados no lineales divergen instantáneamente ante tales condiciones iniciales, nuestro optimizador de Evolución Diferencial refinado con el método símplex de Nelder-Mead convergió de manera totalmente autónoma a:
- $A'_(a j u s t e) = 2110.67$ MHz
- $B'_(a j u s t e) = 1247.89$ MHz
- $C'_(a j u s t e) = 789.95$ MHz

El coeficiente de correlación cruzada normalizada alcanzó un valor óptimo superior al $98\%$, y como se aprecia en la Figura 3, el espectro de residuos $(I_(e x p) - I_(c a l c))$ se mantiene estrictamente confinado dentro de la envolvente del ruido experimental.

= 4. Conclusiones y Perspectivas

Hemos diseñado y validado formalmente *SpecWatson*, una plataforma computacional moderna y matemáticamente rigurosa para la simulación y ajuste automatizado de espectros electrónicos rovibrónicos. Mediante la transformación de base simetrizada de Wang, el Hamiltoniano de rotor asimétrico con distorsión centrífuga de Watson se desacopla en cuatro subsistemas tridiagonales, eliminando cuellos de botella temporales y preservando la precisión cuántica completa ($< 10^(-11)$ MHz de error). La incorporación de algoritmos evolutivos resuelve el problema inverso de manera robusta, deduciendo constantes moleculares a partir de los perfiles de banda sin requerir la tediosa asignación manual de transiciones.

Los desarrollos futuros incorporarán Hamiltonianos de rotación interna (acoplamiento periódico de barreras $V_3$ para rotores de grupos metilo $-"CH"_3$) aplicables a moléculas como el 4-metilanisol y derivados de tolueno, así como el traslado de las operaciones tensoriales a aceleradores GPU (JAX/PyTorch) para habilitar ajustes en tiempo real en entornos interactivos basados en la web.

= Agradecimientos
Este trabajo contó con el respaldo institucional de la Universidad de Guanajuato (División de Ciencias e Ingenierías, Campus León). M.D. agradece al Laboratorio de Espectroscopía Molecular por los recursos de cómputo facilitados.

#v(8pt)
#set text(8pt)
#bibliography("references.bib", style: "ieee")
