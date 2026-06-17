# Simulador de interferometro de Michelson

Programa base en Python para simular el patron de interferencia observado por
una camara en un interferometro de Michelson con haces colimados. El nucleo
fisico esta separado de la salida grafica para que luego sea sencillo agregar
una interfaz visual.

## Modelo fisico

La diferencia de camino optico sobre la camara se calcula como

```text
OPD(x, y) = OPD_central
            + 2 * (tilt_rel_azimutal * x + tilt_rel_cenital * y)
```

donde

```text
tilt_rel_azimutal = mirror_2_azimuth - mirror_1_azimuth
tilt_rel_cenital  = mirror_2_cenital  - mirror_1_cenital
OPD_central       = opd0 + 2 * mirror_2_displacement
```

El factor 2 aparece dos veces por la misma razon fisica: la reflexion duplica
el cambio angular del haz, y el desplazamiento axial de un espejo cambia el
camino optico en ida y vuelta.

Para una fuente monocromatica:

```text
I = I1 + I2 + 2 sqrt(I1 I2) cos(2 pi OPD / lambda)
```

Para una fuente de espectro amplio:

```text
I = I1 + I2 + 2 sqrt(I1 I2) Re[ gamma(OPD) ]
```

`gamma(OPD)` es la coherencia compleja de la fuente. Su fase produce las
franjas y su magnitud produce la envolvente de visibilidad.

## Variables manipulables

Fuente:

- `--source`: `mono`, `gaussian` o `csv`.
- `--wavelength-nm`: longitud de onda central.
- `--fwhm-nm`: ancho espectral FWHM para fuente gaussiana.
- `--spectrum-csv`: archivo con columnas `wavelength_nm,weight`.

Espejos:

- `--mirror-1-azimuth-urad`: inclinacion azimutal del espejo 1.
- `--mirror-1-cenital-urad`: inclinacion cenital del espejo 1.
- `--mirror-2-azimuth-urad`: inclinacion azimutal del espejo 2.
- `--mirror-2-cenital-urad`: inclinacion cenital del espejo 2.
- `--mirror-2-displacement-um`: desplazamiento axial del espejo 2.
- `--opd0-um`: diferencia fija de camino optico antes de desplazar el espejo.

Alias relativos:

- `--tilt-azimuth-urad`: fija directamente `mirror_2_azimuth - mirror_1_azimuth`.
- `--tilt-cenital-urad`: fija directamente `mirror_2_cenital - mirror_1_cenital`.
- `--tilt-cenith-urad`: alias equivalente, conservado por compatibilidad.

Camara y visualizacion:

- `--pixels-x`, `--pixels-y`: resolucion numerica.
- `--camera-width-mm`, `--camera-height-mm`: campo fisico de vision.
- `--zoom`: zoom numerico sobre el campo de vision.
- `--profile-samples`: puntos usados en el perfil perpendicular a las franjas.

Haces:

- `--intensity-1`, `--intensity-2`: intensidades relativas.
- `--contrast`: contraste experimental adicional entre 0 y 1.

## Uso rapido

```powershell
python -m michelson.cli --source gaussian --wavelength-nm 632.8 --fwhm-nm 10 --mirror-2-azimuth-urad 2500 --mirror-2-cenital-urad 500 --output-dir outputs/demo
```

Archivos generados:

- `pattern.png`: patron normalizado de la camara.
- `profile.csv`: perfil perpendicular a las franjas con OPD, intensidad y visibilidad.
- `profile.png`: grafica rapida del perfil.
- `simulation.npz`: arreglos completos para analisis posterior.
- `summary.txt`: parametros y comparacion de longitud de coherencia.

## Interfaz grafica

La interfaz grafica se abre con:

```powershell
python -m michelson.gui
```

Desde la ventana se pueden modificar:

- fuente monocromatica, gaussiana o espectro CSV;
- inclinacion azimutal/cenital de ambos espejos;
- desplazamiento axial del espejo 2;
- OPD fija, zoom, campo de camara, resolucion, intensidades y contraste.

La vista superior muestra el patron de la camara. La linea cyan indica por
donde se toma el perfil perpendicular a las franjas. La vista inferior muestra
la intensidad, la visibilidad teorica y la visibilidad estimada por
maximos/minimos locales.

## Ejemplos

Fuente monocromatica ideal con inclinacion relativa directa:

```powershell
python -m michelson.cli --source mono --wavelength-nm 632.8 --tilt-azimuth-urad 100 --tilt-cenital-urad 0 --output-dir outputs/mono
```

Ambos espejos inclinados: solo importa la diferencia relativa para las franjas.
Aqui el espejo 1 esta en 300 urad y el espejo 2 en 2800 urad, por tanto la
inclinacion relativa azimutal es 2500 urad.

```powershell
python -m michelson.cli --mirror-1-azimuth-urad 300 --mirror-1-cenital-urad 100 --mirror-2-azimuth-urad 2800 --mirror-2-cenital-urad 600 --output-dir outputs/two_mirrors
```

Desplazamiento axial del espejo 2. Moverlo 5 um cambia la OPD central en 10 um:

```powershell
python -m michelson.cli --mirror-2-displacement-um 5 --output-dir outputs/displaced
```

Fuente gaussiana de espectro ancho:

```powershell
python -m michelson.cli --source gaussian --wavelength-nm 632.8 --fwhm-nm 20 --opd0-um 0 --tilt-azimuth-urad 4500 --tilt-cenital-urad 0 --output-dir outputs/gaussian_20nm
```

Zoom optico de la camara simulada:

```powershell
python -m michelson.cli --zoom 3 --output-dir outputs/zoom_3
```

## Longitud de coherencia

Para espectros gaussianos, el programa reporta la longitud de coherencia como
el FWHM de la envolvente de visibilidad en OPD:

```text
Lc = (4 ln 2 / pi) * lambda0^2 / Delta_lambda
```

Tambien reporta el FWHM obtenido numericamente desde la visibilidad calculada
en el perfil. Si el campo de vision o la inclinacion no cubren suficiente rango
de OPD, el resumen lo indica.

## Espectro personalizado

Se puede cargar un CSV con dos columnas:

```text
wavelength_nm,weight
620,0.2
630,1.0
640,0.2
```

Ejemplo:

```powershell
python -m michelson.cli --source csv --spectrum-csv fuente.csv --output-dir outputs/fuente
```

## Pruebas

```powershell
python -m unittest discover -s tests
```
