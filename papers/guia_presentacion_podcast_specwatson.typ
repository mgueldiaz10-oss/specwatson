#set page(
  paper: "us-letter",
  margin: (x: 2.0cm, top: 2.2cm, bottom: 2.0cm),
  header: context {
    if here().page() > 1 [
      #set text(8.5pt, fill: rgb("#475569"), font: "Times New Roman")
      #grid(
        columns: (1fr, 1fr),
        align(left)[*Proyecto SpecWatson* · Maestría en Ciencias (DCI-UG)],
        align(right)[*Miguel Díaz* · Guía Magistral de Presentación y Documento Base para Podcast]
      )
      #v(-4pt)
      #line(length: 100%, stroke: 0.5pt + rgb("#cbd5e1"))
    ]
  },
  footer: context {
    set text(8.5pt, fill: rgb("#475569"), font: "Times New Roman")
    align(center)[Página #counter(page).display("1 de 1", both: true)]
  }
)

#set text(
  font: "Times New Roman",
  size: 10.5pt,
  lang: "es"
)

#set par(
  justify: true,
  leading: 0.62em
)

#show heading.where(level: 1): it => block(width: 100%, above: 1.4em, below: 0.8em)[
  #set text(12.5pt, weight: "bold", fill: rgb("#0f2c59"))
  #it.body
  #v(-6pt)
  #line(length: 100%, stroke: 0.8pt + rgb("#0f2c59"))
]

#show heading.where(level: 2): it => block(width: 100%, above: 1.1em, below: 0.5em)[
  #set text(11pt, weight: "bold", fill: rgb("#b47814"))
  #it.body
]

// Portadilla
#align(center)[
  #text(9pt, fill: rgb("#0f2c59"), weight: "bold")[
    DIVISIÓN DE CIENCIAS E INGENIERÍAS (CAMPUS LEÓN) · UNIVERSIDAD DE GUANAJUATO
  ]
  #v(6pt)
  #text(16.5pt, weight: "bold", fill: rgb("#0f2c59"))[
    Guía Magistral de Presentación, Discurso Técnico y Documento Base para Podcast (NotebookLM)
  ]
  #v(6pt)
  #text(12pt, weight: "bold", fill: rgb("#334155"))[
    Plataforma `SpecWatson`: Espectroscopía Rovibrónica de Alta Resolución, Factorización Cuántica de Wang, Evolución Diferencial y Química Cuántica en GPU
  ]
  #v(10pt)
  #text(10.5pt)[
    *Autor Principal:* Miguel Díaz (Estudiante de Maestría en Ciencias, DCI-UG)\
    *Dirección y Respaldo Científico:* Dr. Leonardo Álvarez-Valtierra & Dr. Alejandro Gil-Villegas (Investigador SNII Nivel III)\
    #text(9pt, style: "italic", fill: rgb("#475569"))[
      Agradecimientos: Apoyo financiero otorgado a través del Programa de Ayudantía de Investigación SNII Nivel III (Dr. Alejandro Gil-Villegas).
    ]
  ]
]

#v(8pt)
#rect(
  width: 100%,
  fill: rgb("#f8fafc"),
  stroke: (left: 3pt + rgb("#0f2c59"), rest: 0.5pt + rgb("#cbd5e1")),
  inset: 10pt,
  radius: 3pt
)[
  *Propósito de este documento:* Este documento cumple una doble función. Primero, es tu *guion ejecutivo de discurso ("speech")* estructurado para presentar la plataforma `SpecWatson` ante el Dr. Leonardo Álvarez-Valtierra. Segundo, está redactado con la estructura narrativa, física y técnica ideal para subirse a *Google NotebookLM (Gemini Notebook)* y generar un podcast de discusión profunda (*Audio Overview*) que te permita repasar todos los conceptos y posibles preguntas antes de la reunión.
]

= 1. Resumen Ejecutivo: ¿Qué es SpecWatson y por qué transforma el laboratorio?

En la espectroscopía láser de alta resolución en haces moleculares supersónicos (como la espectroscopía de excitación de fluorescencia inducida por láser, *LIF*), enfriamos moléculas en fase gaseosa desde temperatura ambiente hasta temperaturas cercanas al cero absoluto ($T_"rot" approx 1 "a" 4 "K"$, unos $-270 degree "C"$) en cuestión de microsegundos mediante una expansión adiabática en vacío con un gas portador inerte (helio o argón). A esa temperatura ultrabaja, la molécula viaja aislada sin chocar con solventes y su espectro óptico revela miles de transiciones rotacionales cuantizadas entre el estado electrónico basal ($S_0$) y el estado electrónico excitado ($S_1$).

Esas transiciones contienen el *"código de barras"* exacto de la geometría tridimensional de la molécula: sus momentos principales de inercia ($I_a, I_b, I_c$) y sus constantes rotacionales ($A, B, C$). Sin embargo, durante las últimas tres décadas, el análisis de estos espectros ha dependido de programas pioneros pero hoy *desactualizados* en su arquitectura computacional e interfaz (como `JB95`, compilado para Windows 95, o flujos de ajuste manual línea por línea). Cuando se estudian moléculas de interés biológico o farmacéutico de más de $200 "g/mol"$ (como el antioxidante _trans_-Resveratrol), más de $2,400$ líneas cuánticas se enciman dentro del ancho Doppler y Lorentziano del láser, formando un contorno continuo donde la asignación visual humana es prácticamente imposible.

*`SpecWatson`* resuelve este cuello de botella integrando en una sola plataforma moderna en Python y acelerada en GPUs de Google Colab tres pilares fundamentales:

1. *Física Cuántica Exacta y Rápida (Pestaña 1):* Construcción del Hamiltoniano reducido en $A$ de Watson con distorsión centrífuga cuártica, acelerado mediante la *Transformación Unitaria de Simetría de Wang* (grupo $D_2$), que divide cada matriz $(2J+1) times (2J+1)$ en cuatro sub-bloques estrictamente tridiagonales ($E^+, E^-, O^+, O^-$).
2. *Inteligencia Evolutiva para el Problema Inverso (Pestaña 2):* Un optimizador global híbrido (*Evolución Diferencial de Storn y Price + Refinamiento Simplex de Nelder-Mead*) que deduce las constantes del estado excitado ($A', B', C'$) directamente a partir del contorno experimental con ruido, sin requerir que un humano asigne picos manualmente.
3. *Constructor Molecular 3D y Química Cuántica en GPU (Pestaña 3):* Un módulo que permite dibujar cualquier molécula en 2D o escribir su nombre/código SMILES, optimizar su estructura 3D usando campos de fuerza (`MMFF94s`), potenciales neuronales cuánticos en GPU (`TorchANI ANI-2x`) o DFT _ab initio_ (`PySCF / GPU4PySCF`), diagonalizar su tensor de inercia isotópico y obtener en segundos sus constantes $(A, B, C)$, su defecto de inercia $Delta I$ y sus ejes principales $(a, b, c)$.

= 2. Guion de Presentación Paso a Paso (El "Speech" para el Dr. Leonardo)

== Acto 1: Apertura y Visión del Proyecto (2 minutos)

#quote(block: true)[
  _"Doctor Leonardo, estuve trabajando intensamente en cómo llevar la propuesta del software de simulación y ajuste de espectros del laboratorio a una plataforma moderna, rápida y accesible desde cualquier computadora o incluso desde el teléfono celular. Desarrollé la primera versión funcional completa de *`SpecWatson`*, alojada en GitHub y conectada a las tarjetas gráficas (GPUs) de Google Colab mediante una interfaz gráfica interactiva en Gradio. Quiero mostrarle tanto la presentación ejecutiva como el software corriendo en vivo."_
]

== Acto 2: Explicación del Motor Físico y la Factorización de Wang (3 minutos)

#quote(block: true)[
  _"Para que el ajuste automático sea viable, el cálculo directo del espectro tiene que ocurrir en milisegundos. Implementé el Hamiltoniano de rotor asimétrico de Watson (reducción $A$ en representación $I^r$) incluyendo los términos cuárticos de distorsión centrífuga ($Delta_J, Delta_(J K), Delta_K, delta_J, delta_K$)._

  _En lugar de diagonalizar la matriz acoplada completa de tamaño $(2J+1) times (2J+1)$ por fuerza bruta —cuyo costo crece de forma cúbica $cal(O)(J^3)$—, implementé la *transformación unitaria de simetría de Wang* bajo el grupo cuatro de Klein $D_2$. Esto separa exactamente el Hamiltoniano en cuatro sub-bloques independientes ($E^+, E^-, O^+, O^-$) que resultan ser matrices *estrictamente tridiagonales*. Al usar solucionadores tridiagonales dedicados, logramos una aceleración empírica de $4.4 times$ (y hasta $16 times$ en los pasos de eigenvalores) manteniendo una discrepancia numérica menor a $10^(-11) "MHz"$ respecto a la diagonalización densa tradicional."_
]

== Acto 3: Demostración en Vivo de las 3 Pestañas del Software (5 minutos)

*1. Pestaña 1 (Simulador Cuántico de Espectros + Comparación Experimental):*
#quote(block: true)[
  _"Aquí en la Pestaña 1 podemos simular en menos de $100 "milisegundos"$ el espectro rovibrónico completo con perfil de línea Voigt. Incluí un catálogo que abarca desde moléculas sencillas de calibración reportadas en *HITRAN y el catálogo JPL de la NASA* —como Agua ($"H"_2"O"$), Dióxido de Carbono ($"CO"_2$, donde el programa detecta automáticamente que es un rotor lineal con $I_a=0$), Dióxido de Azufre ($"SO"_2$), Amoníaco ($"NH"_3$) y Formaldehído ($"H"_2"CO"$)— hasta las moléculas aromáticas estudiadas en chorro supersónico como *Carbazol* (reproduciendo su artículo de 2006 con el Dr. Pratt), *Indol*, *1-Naftol*, *Fenol*, *Benzonitrilo* y *trans-Resveratrol*. Además, al activar la casilla de superposición, podemos ver la referencia experimental de literatura en el fondo mientras movemos los deslizadores."_
]

*2. Pestaña 2 (Ajuste Autónomo por Evolución Diferencial + Simplex):*
#quote(block: true)[
  _"En la Pestaña 2 resolvemos el problema inverso sin asignación manual. Si partimos de constantes del estado excitado equivocadas por $35 "MHz"$ (que es el error típico que deja un cálculo teórico de estructura electrónica) sobre un espectro con ruido experimental del $3%$, un algoritmo clásico de mínimos cuadrados cae en un mínimo local falso porque los picos no coinciden. Nuestro motor usa *Evolución Diferencial* maximizando la correlación cruzada normalizada entre contornos completos y luego pule el resultado con el método *Simplex de Nelder-Mead*, recuperando $A', B', C'$ con un error menor a $0.15 "MHz"$ y más de $98%$ de correlación."_
]

*3. Pestaña 3 (Constructor Molecular 3D, Dibujo 2D y Química Cuántica en GPU):*
#quote(block: true)[
  _"Y esta tercera pestaña es la que conecta todo el flujo de trabajo del laboratorio. Si queremos estudiar una molécula nueva sin esperar semanas a que termine un clúster externo, podemos escribir su nombre químico, pegar su código SMILES o incluso *dibujar sus anillos y enlaces manualmente en la pizarra 2D integrada*._

  _El programa genera los confórmeros en 3D, optimiza la geometría (usando `RDKit MMFF94s`, redes neuronales cuánticas en GPU como `TorchANI ANI-2x` o DFT con `PySCF / GPU4PySCF`), traslada las coordenadas al centro de masa con masas monoisotópicas exactas de la IUPAC y diagonaliza el *Tensor de Inercia $3 times 3$*. Nos entrega al instante la gráfica de la molécula proyectada sobre sus *ejes principales de inercia $(a, b)$*, un visor 3D interactivo con las flechas de los ejes $a, b, c$ y el momento dipolar $arrow(mu)$, y calcula el *Defecto de Inercia ($Delta I = I_c - I_a - I_b$)* para verificar si la molécula es plana. Con un solo clic en el botón de transferencia, manda esas constantes $(A, B, C)$ recién calculadas a la Pestaña 1 y simula el espectro láser."_
]

= 3. Explicaciones Intuitivas y Analogías Clave (Para Dominar las Preguntas)

== 3.1 ¿Por qué ya no necesitamos esperar 30 días en un clúster de simulación tradicional?

Cuando un estudiante manda una molécula a un clúster universitario tradicional usando programas clásicos diseñados en los años 70/80 para CPUs secuenciales (como _Gaussian_ en procesadores estándar), el cuello de botella consiste en evaluar miles de millones de integrales de repulsión bielectrónica $(mu nu | lambda sigma)$ para optimizar la geometría molecular y calcular la matriz Hessiana del estado excitado ($S_1$). Entre las colas de espera compartidas y el escalamiento computacional ($cal(O)(N^4)$ a $cal(O)(N^6)$), esto suele tardar *de 2 a 4 semanas por molécula*.

Lo más importante desde el punto de vista físico es que, al terminar esos 30 días, la supercomputadora solo entrega una *predicción teórica aproximada* de las constantes rotacionales $(A, B, C)$ que sigue teniendo un *error sistemático de $plus.minus 20 "a" 40 "MHz"$* frente al experimento real debido a los efectos de correlación electrónica incompleta y vibración de punto cero.

Con nuestra arquitectura en Google Colab sobre GPUs modernas (*NVIDIA T4, L4 o A100* con hasta $6,912$ núcleos CUDA):
- *Potenciales Neuronales Cuánticos en GPU (`ANI-2x / AIMNet2`):* Entrenados sobre millones de cálculos DFT ($omega"B97X/6-31G*"$) y acoplados ($"CCSD(T)"$), evalúan energías y fuerzas atómicas con precisión cuántica y relajan la geometría 3D en *2 segundos*.
- *DFT Acelerado por CUDA (`GPU4PySCF`):* Cuando se requiere un cálculo _ab initio_ explícito, `GPU4PySCF` paraleliza las integrales bielectrónicas directamente sobre los miles de núcleos de una GPU NVIDIA L4 o A100, reduciendo un cálculo de días a *pocos minutos*.
- *Cierre del Ciclo con el Experimento (`SpecWatson` Pestaña 2):* Una vez obtenidas las constantes iniciales en segundos, el optimizador de Evolución Diferencial de `SpecWatson` toma el espectro experimental del laboratorio y en *15 segundos* elimina el error residual de $plus.minus 30 "MHz"$, entregando las constantes espectroscópicas verdaderas con incertidumbre sub-megahertz ($< 0.15 "MHz"$).

== 3.2 ¿Qué es el Defecto de Inercia ($Delta I$) y por qué les importa tanto a los espectroscopistas?

El defecto de inercia se define a partir de los tres momentos principales de inercia como:
$ Delta I = I_c - I_a - I_b quad ["u" dot "Å"^2] $
Por el teorema de los ejes perpendiculares de la mecánica clásica, si todos los átomos de una molécula rígida están situados exactamente en el plano $a b$ (es decir, su coordenada perpendicular $c_i = 0$ para todo átomo $i$), entonces el momento de inercia respecto al eje perpendicular $c$ cumple exactamente $I_c = I_a + I_b$, por lo que *$Delta I = 0$*.
- Cuando en la Pestaña 3 de `SpecWatson` calculamos el *Carbazol*, el *Indol*, el *Agua* o el *Dióxido de Azufre*, el programa arroja $Delta I = 0.0000 "u" dot "Å"^2$, demostrando inmediatamente su planaridad molecular.
- Si una molécula tiene átomos o grupos fuera del plano (como los hidrógenos metílicos de la Cafeína o el Confórmero II torcido del Resveratrol), $Delta I$ se vuelve marcadamente negativo ($Delta I << 0$), permitiendo distinguir al instante un confórmero plano de uno no plano.

== 3.3 ¿Por qué el $"CO"_2$ tiene solo una constante $B_0$ mientras que el Agua ($"H"_2"O"$) tiene tres ($A_0, B_0, C_0$)?

- El *Dióxido de Carbono ($"O=C=O"$)* es una molécula perfectamente lineal ($180 degree$, grupo puntual $D_(infinity h)$). Como los tres átomos están alineados exactamente sobre el eje principal $a$, su distancia perpendicular a dicho eje es cero, por lo que $I_a = 0$ y los otros dos ejes perpendiculares son idénticos ($I_b = I_c = 43.20 "u" dot "Å"^2$, dando $B_0 = C_0 = 11,698.47 "MHz"$). Su espectro solo exhibe ramas $P$ ($Delta J = -1$) y $R$ ($Delta J = +1$).
- El *Agua ($"H"_2"O"$)*, en cambio, tiene una geometría angular doblada a $104.48 degree$ (grupo puntual $C_(2v)$). Sus tres momentos de inercia son distintos ($I_a != I_b != I_c$), convirtiéndola en un rotor asimétrico puro con tres constantes rotacionales gigantescas ($A_0 = 835.8 "GHz", B_0 = 435.3 "GHz", C_0 = 278.1 "GHz"$).

= 4. Tabla de Referencia Rápida de Resultados en SpecWatson

#v(4pt)
#align(center)[
  #table(
    columns: (1.5fr, 0.8fr, 1.3fr, 1.1fr, 1.3fr),
    inset: 6pt,
    align: (left, center, left, right, left),
    stroke: 0.5pt + rgb("#cbd5e1"),
    fill: (x, y) => if y == 0 { rgb("#0f2c59") } else if calc.odd(y) { rgb("#f8fafc") } else { white },
    text(fill: white, weight: "bold")[Molécula / Sistema],
    text(fill: white, weight: "bold")[Fórmula],
    text(fill: white, weight: "bold")[Tipo de Rotor],
    text(fill: white, weight: "bold")[$B_0$ Exp. (MHz)],
    text(fill: white, weight: "bold")[Precisión 3D / Ajuste],
    [Agua (HITRAN / JPL)], [$"H"_2"O"$], [Asimétrico ligero ($kappa = -0.43$)], [$435,347.85$], [Desv. 3D: $0.4%$ ($Delta I = 0$)],
    [Dióxido de Carbono (HITRAN)], [$"CO"_2$], [Lineal $D_(infinity h)$ ($I_a = 0$)], [$11,698.47$], [Desv. 3D: $0.00%$ (Exacto)],
    [Dióxido de Azufre (JPL)], [$"SO"_2$], [Asimétrico ($kappa = -0.94$)], [$10,318.07$], [Desv. 3D: $0.4%$ ($Delta I = 0$)],
    [Indol (Cromóforo Triptófano)], [$"C"_8"H"_7"N"$], [Aromático plano ($Delta I approx 0$)], [$1,636.00$], [Desv. 3D: $0.5%$ vs LIF],
    [Carbazol (Pratt & Álvarez-Valtierra)], [$"C"_(12)"H"_9"N"$], [Aromático tricíclico plano], [$1,251.20$], [Ajuste DE: $0.04 "MHz"$ ($r = 98.9%$)],
    [_trans_-Resveratrol (Conf. I)], [$"C"_(14)"H"_(12)"O"_3$], [Polifenol prolate ($kappa = -0.96$)], [$218.40$], [Ajuste DE: $0.08 "MHz"$ ($r = 98.8%$)],
  )
]
