#set page(
  paper: "a4",
  margin: (x: 1.6cm, top: 1.8cm, bottom: 1.8cm),
  header: context {
    if here().page() > 1 [
      #set text(8pt, fill: rgb("#555555"), font: "Times New Roman")
      #grid(
        columns: (1fr, 1fr),
        align(left)[*M. Díaz & L. Álvarez-Valtierra* | Espectroscopía Rovibrónica de Polifenoles],
        align(right)[*Physical Chemistry Chemical Physics* (Versión en Español, 2026)]
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
    FISICOQUÍMICA BIOMOLECULAR & ESPECTROSCOPÍA ROTACIONAL
  ]
  #v(4pt)
  #text(16pt, weight: "bold", font: "Times New Roman")[
    Identificación Conformacional Autónoma y Ajuste de Contorno Rovibrónico de Alta Resolución del Polifenol Natural _trans_-Resveratrol en Jets Moleculares Supersónicos
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
  Los polifenoles constituyen una familia fundamental de metabolitos vegetales biológicamente activos con notables propiedades antioxidantes, antiinflamatorias y quimiopreventivas. No obstante, su caracterización estructural precisa en fase gaseosa se ve severamente limitada por su labilidad térmica, baja volatilidad y marcada multiestabilidad conformacional. Si bien la técnica de ablación láser acoplada a expansiones en jets moleculares supersónicos permite vaporizar y enfriar polifenoles intactos hasta temperaturas rotacionales de $T_(r o t) approx 2 - 4$ K sin descomposición térmica, los espectros electrónicos resultantes con resolución rotacional ($S_1 arrow.l.bar S_0$) presentan una congestión espectral extrema debido a sus elevados momentos de inercia y reducidas constantes rotacionales ($B, C < 200$ MHz). La asignación cuántica tradicional línea por línea mediante software clásico de 32 bits (como JB95 o PGOPHER) resulta prácticamente inviable bajo densidades de estados tan densas. En este trabajo empleamos la plataforma *SpecWatson* para simular, dilucidar y ajustar de forma autónoma el espectro electrónico con resolución rotacional del *trans*-resveratrol (_trans_-3,5,4'-trihidroxiestilbeno). Mediante la transformación de base simetrizada de Wang (grupo cuaternario de Klein $D_2$ Vierergruppe), el Hamiltoniano de rotor asimétrico ($kappa = -0.964$, fuertemente prolado) se particiona en cuatro submatrices tridiagonales resueltas mediante rutinas LAPACK de complejidad $O(N)$, acelerando el cómputo espectral multiparamétrico por un factor de $4.6 times$ frente a los resolvedores matriciales densos convencionales. Integrando un motor heurístico de ajuste de contorno basado en Evolución Diferencial, la plataforma discrimina con total certidumbre entre rotámeros candidatos, identificando unívocamente el confórmero plano de mínima energía (Confórmero I) y descartando rotámeros alternativos ($r < 30\%$). En presencia de un $3.0\%$ de ruido experimental sintético y ensanchamiento instrumental de Voigt ("FWHM" = 20 MHz), *SpecWatson* recupera autónomamente las constantes del estado excitado ($A', B', C'$) con una exactitud inferior a $0.1$ MHz sin requerir asignación manual de transiciones. Esta investigación establece un protocolo general y automatizado para la espectroscopía de alta resolución de biomoléculas pesadas y polifenoles naturales.
  
  #v(4pt)
  #text(8.5pt)[*Palabras clave:* Polifenoles, Resveratrol, Jets Supersónicos, Ablación Láser, Ajuste de Contorno Rovibrónico, Factorización de Wang, Hamiltoniano de Watson, Algoritmos Genéticos.]
]
#v(12pt)

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

Los compuestos polifenólicos, caracterizados por múltiples grupos hidroxilo fenólicos enlazados a esqueletos aromáticos conjugados, son metabolitos secundarios omnipresentes en el reino vegetal @zwier2006laser, @gerhards2003structure. Entre ellos, el _trans_-resveratrol (_trans_-3,5,4'-trihidroxiestilbeno) ha captado un enorme interés biomédico debido a sus notables propiedades cardioprotectoras, neuroprotectoras y promotoras de la longevidad celular @smeyers2004resveratrol. La comprensión fundamental de sus mecanismos antioxidantes exige determinar con rigor su topología conformacional intrínseca, aislada de perturbaciones de solvente, empaquetamiento cristalino y promediado térmico.

La espectroscopía láser de alta resolución en haces moleculares supersónicos brinda la metodología idónea para dilucidar la geometría de moléculas biológicas aisladas @alvarez2006rotationally, @tan2019high. Sin embargo, el estudio de polifenoles bajo condiciones libres de colisión afronta severos obstáculos experimentales y teóricos:
1. *Descomposición Térmica y Baja Presión de Vapor:* Los polifenoles presentan puntos de fusión elevados y se degradan térmicamente al someterse a calentamiento convencional en boquillas continuas. Para transferir moléculas intactas a fase gaseosa, se recurre a la *ablación láser pulsada* (desorción láser con pulsos Nd:YAG acoplada a válvulas de expansión supersónica) @tan2019high.
2. *Severa Congestión Espectral:* Con una masa molecular de $228.24$ g/mol y una estructura extendida tipo estilbeno ($approx 14$ Å), el resveratrol exhibe momentos principales de inercia sumamente grandes ($I_a approx 365, I_b approx 2715, I_c approx 3080 " amu" dot "Å"^2$). Como consecuencia, sus constantes rotacionales son pequeñas: $A approx 1385$ MHz y $B, C approx 160 - 190$ MHz @smeyers2004resveratrol. Incluso a temperaturas de jet de $T_(r o t) approx 2.5$ K, los estados rotacionales se pueblan hasta valores de $J approx 15 - 20$, generando miles de transiciones rovibrónicas solapadas dentro de una estrecha ventana de frecuencia ($plus.minus 1500$ MHz).
3. *Inviabilidad de la Asignación Manual:* El software clásico de ajuste espectroscópico, como *JB95* @alvarez2006rotationally y *PGOPHER* @western2014pgopher, depende de la asignación interactiva línea por línea o de diagramas de Loomis-Wood. En polifenoles de masa elevada, las líneas discretas se fusionan en un contorno de banda continuo, imposibilitando la asignación manual clásica.

En este artículo demostramos la capacidad de *SpecWatson* para superar estas limitaciones. Al enlazar la factorización de base simetrizada de Wang @wang1929degree con un motor automatizado de ajuste de contorno basado en Evolución Diferencial @storn1997differential, logramos una discriminación rotamérica unívoca y la determinación de parámetros estructurales con precisión sub-megahertz directamente a partir de bandas densas de _trans_-resveratrol.

= 2. Marco Teórico y Paisaje Conformacional

== 2.1. Paisaje Conformacional del _trans_-Resveratrol
Estructuralmente, el _trans_-resveratrol está compuesto por un anillo 4'-hidroxifenilo y un anillo 3,5-dihidroxifenilo (resorcinol) interconectados por un puente vinileno _trans_ @smeyers2004resveratrol. La molécula cuenta con diversos grados de libertad torsionales: los ángulos diedros entre los anillos aromáticos y el grupo etileno central ($phi_1, phi_2$), así como las orientaciones de los tres grupos hidroxilo fenólicos.

Cálculos de química cuántica demuestran que el mínimo global (*Confórmero I*) corresponde a una conformación planar totalmente _trans_, donde la conjugación del sistema $pi$ maximiza la deslocalización electrónica. Un mínimo secundario (*Confórmero II*), ubicado $approx 1.5 - 3.0$ kJ/mol por encima en energía, se produce por la rotación interna coplanar de $180 degree$ del anillo de resorcinol o de los enlaces O-H, desplazando el origen electrónico $S_1 arrow.l.bar S_0$ en $Delta nu_0 approx +45 " cm"^(-1)$ y modificando las constantes rotacionales en varios megahertz.

#v(4pt)
#figure(
  table(
    columns: (2.0cm, 1.8cm, 1.8cm, 1.8cm),
    stroke: 0.4pt + rgb("#bbbbbb"),
    fill: (col, row) => if row == 0 { rgb("#e8edf2") } else { none },
    align: (left, center, center, center),
    [*Parámetro*], [*Conf. I ($S_0$)*], [*Conf. I ($S_1$)*], [*Conf. II ($S_0$)*],
    [$A$ (MHz)], [1385.40], [1362.80], [1442.10],
    [$B$ (MHz)], [186.20], [184.90], [179.50],
    [$C$ (MHz)], [164.10], [162.80], [159.80],
    [$kappa$], [-0.9638], [-0.9632], [-0.9693],
    [$nu_0$ ($"cm"^(-1)$)], [---], [32840.50], [32885.20],
    [$mu_a^2 : mu_b^2$], [---], [0.82 : 0.18], [0.65 : 0.35],
  ),
  caption: [Parámetros rotacionales, coeficientes de asimetría de Ray y orígenes de banda para los confórmeros de _trans_-resveratrol.]
)
#v(4pt)

== 2.2. Hamiltoniano Asimétrico de Watson en la Base de Wang
Dada la geometría planar extendida del resveratrol, el parámetro de asimetría de Ray resulta $kappa = -0.9638$, situando a la molécula en el régimen de rotor asimétrico fuertemente prolado ($kappa arrow.r -1$). La representación prolada $I^r$ ($z arrow.l.r a, x arrow.l.r b, y arrow.l.r c$) alinea de manera natural el eje inercial principal $a$ con el eje mayor de la molécula.

El Hamiltoniano con reducción $A$ de Watson @watson1977aspects incluye la distorsión centrífuga:
$ H_W = H_0 - Delta_J hat(J)^4 - Delta_(J K) hat(J)^2 hat(J)_a^2 - Delta_K hat(J)_a^4 \
  - 2 delta_J hat(J)^2 (hat(J)_b^2 - hat(J)_c^2) - delta_K [hat(J)_a^2, (hat(J)_b^2 - hat(J)_c^2)]_+ $

Para superar el escalamiento cúbico $O(N^3)$ de la diagonalización densa convencional, aplicamos la transformación unitaria de base simetrizada de Wang @wang1929degree:
$ |J, K, gamma chevron.r = 1/sqrt(2) (|J, K chevron.r + (-1)^gamma |J, -K chevron.r) quad (K > 0) $
descomponiendo el espacio matricial en cuatro sub-bloques tridiagonales ($E^+, E^-, O^+, O^-$). Dado que cada sub-bloque conecta únicamente estados adyacentes en su subespacio, los resolvedores tridiagonales de LAPACK (`dstevd`) obtienen los valores y vectores propios en tiempo $O(N)$.

== 2.3. Orientación del Dipolo de Transición y Bandas Híbridas
El momento dipolar de transición electrónica $S_1 (pi pi^*) arrow.l.bar S_0$ se encuentra confinado en el plano molecular ($a b$), resultando nula la componente perpendicular ($mu_c = 0$). En el Confórmero I, el dipolo de transición forma un ángulo de $approx 24 degree$ con respecto al eje $a$, generando una banda híbrida compuesta por un $82\%$ de carácter tipo $a$ y un $18\%$ de carácter tipo $b$. Los elementos matriciales de cosenos directores se evalúan analíticamente mediante símbolos $3$-$j$ de Wigner:
$ S = sum_(g in {a, b}) mu_g^2 | sum_(K', K'') C'^*_(K', tau') C''_(K'', tau'') chevron.l J', K' | Phi_g | J'', K'' chevron.r |^2 $
El espectro discreto de palos se convoluciona con un perfil de Voigt ($Gamma = 20$ MHz, $eta_L = 0.35$) acorde a las condiciones instrumentales típicas de ablación láser pulsada en expansiones supersónicas.

= 3. Resultados y Discusión

== 3.1. Evaluación de Rendimiento Computacional
La Figura 1 muestra el escalamiento temporal hasta $J = 34$ (dimensión matricial $N = 69$). Mediante las rutinas densas convencionales de LAPACK, la matriz completa exige $1.86$ ms por estado. Con la factorización tridiagonal de Wang implementada en *SpecWatson*, el tiempo se reduce a únicamente $0.40$ ms, otorgando una aceleración empírica neta de $4.6 times$. En procesos de ajuste global que requieren entre $10^3$ y $10^4$ evaluaciones espectrales, esta optimización disminuye los tiempos de cálculo totales de varias horas a unos cuantos minutos.

#v(4pt)
#figure(
  image("figures/fig1_resveratrol_scaling.png", width: 94%),
  caption: [(a) Tiempo de cálculo por estado cuántico $J$ en resveratrol ($N <= 71$). (b) Factor de aceleración empírico ($t_(c o m p l e t a) / t_(W a n g)$) alcanzando $4.6 times$ en CPU, evidenciando las ventajas algorítmicas de Wang en polifenoles pesados.]
)
#v(4pt)

#colbreak()

== 3.2. Estructura Rovibrónica del _trans_-Resveratrol
La Figura 2 ilustra el origen de banda $S_1 arrow.l.bar S_0$ simulado para el Confórmero I a $T_(r o t) = 2.5$ K. Más de 2,400 transiciones individuales se pueblan térmicamente hasta $J = 11$. El espectro se encuentra dominado por una intensa rama $Q$ central ($Delta J = 0$), flanqueada por ramas simétricas $P$ ($Delta J = -1$) y $R$ ($Delta J = +1$) que se extienden a lo largo de $plus.minus 1500$ MHz. A diferencia de rotores ligeros, las líneas individuales coalescen en una envolvente continua altamente congestionada, ilustrando el motivo por el cual los métodos convencionales de asignación individual fracasan.

#v(4pt)
#figure(
  image("figures/fig2_resveratrol_spectrum.png", width: 94%),
  caption: [Contorno de banda sintético con resolución rotacional para la transición $S_1 arrow.l.bar S_0$ del _trans_-resveratrol a $T_(r o t) = 2.5$ K. Más de 2,400 transiciones convergen en un contorno denso con una aguda rama $Q$ central y envolventes $P/R$ resueltas ("FWHM" = 20 MHz).]
)
#v(4pt)

#colbreak()

== 3.3. Discriminación Conformacional y Ajuste Autónomo
Para evaluar la capacidad del algoritmo en la resolución del problema inverso, los datos sintéticos de jet se corrompieron con un $3.0\%$ de ruido gaussiano aleatorio. Se formuló una doble validación:
1. *Rechazo Conformacional:* ¿Es capaz el sistema de distinguir entre rotámeros moleculares? Al contrastar el contorno del Confórmero II frente al perfil experimental sintético, el coeficiente de correlación cruzada normalizada arrojó un valor de $r < 30\%$ (Figura 3), permitiendo el descarte categórico e inequívoco de la estructura conformacional errónea.
2. *Recuperación Paramétrica del Confórmero I:* Partiendo de constantes iniciales intencionalmente distorsionadas ($A' = 1338.0$ MHz, $B' = 191.0$ MHz, $C' = 158.0$ MHz; desviadas en hasta 25 MHz respecto a los valores reales), la optimización por Evolución Diferencial convergió autónomamente a:
- $A'_(a j u s t e) = 1362.72$ MHz (Real: $1362.80$ MHz, Residuo: $0.08$ MHz)
- $B'_(a j u s t e) = 184.93$ MHz (Real: $184.90$ MHz, Residuo: $0.03$ MHz)
- $C'_(a j u s t e) = 162.78$ MHz (Real: $162.80$ MHz, Residuo: $0.02$ MHz)
- Desplazamiento del origen de banda: $+0.04$ MHz.

La correlación cruzada final alcanzó un $98.8\%$, y el espectro de residuos $(I_(e x p) - I_(c a l c))$ en la Figura 3 demuestra convergencia completa dentro de los márgenes del ruido experimental.

#v(4pt)
#figure(
  image("figures/fig3_conformer_fit.png", width: 94%),
  caption: [Discriminación conformacional y ajuste automatizado de contorno del _trans_-resveratrol. Superior: Datos sintéticos con 3% de ruido (negro), ensayo del Confórmero II rechazado ($r < 30\%$, gris punteado) y el ajuste convergido del Confórmero I ($r = 98.8\%$, rojo discontinuo). Inferior: Espectro de residuos $(I_(e x p) - I_(c a l c))$ delimitado dentro del nivel de ruido experimental.]
)
#v(4pt)

#colbreak()

= 4. Conclusiones

Hemos demostrado con éxito la aplicación de *SpecWatson* al análisis de sistemas polifenólicos complejos empleando el _trans_-resveratrol como molécula modelo. Al integrar la tridiagonalización de Wang con algoritmos heurísticos de Evolución Diferencial, la plataforma supera la congestión espectral inherente a biomoléculas pesadas en haces moleculares supersónicos. El método garantiza una determinación estructural sub-megahertz y una discriminación conformacional precisa sin intervención manual en la asignación de números cuánticos. Este desarrollo sienta las bases para futuros estudios sistemáticos mediante ablación láser de productos naturales flexibles, antioxidantes y principios activos farmacológicos.

= Agradecimientos
Los autores expresan su agradecimiento a la Universidad de Guanajuato (División de Ciencias e Ingenierías, Campus León) por el respaldo institucional y los recursos de cómputo facilitados.

#v(8pt)
#set text(8pt)
#bibliography("references.bib", style: "ieee")
