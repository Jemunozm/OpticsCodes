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

- `--source`: `mono`, `rectangular` o `csv`.
- `--wavelength-nm`: longitud de onda central.
- `--width-nm`: ancho espectral para fuente rectangular uniforme.
- `--spectrum-csv`: archivo con columnas `wavelength_nm,weight`.

Espejos:

- `--mirror-1-azimuth-udeg`: inclinacion azimutal del espejo 1 en microgrados.
- `--mirror-1-cenital-udeg`: inclinacion cenital del espejo 1 en microgrados.
- `--mirror-2-azimuth-udeg`: inclinacion azimutal del espejo 2 en microgrados.
- `--mirror-2-cenital-udeg`: inclinacion cenital del espejo 2 en microgrados.
- `--mirror-2-displacement-um`: desplazamiento axial del espejo 2.
- `--opd0-um`: diferencia fija de camino optico antes de desplazar el espejo.

Alias relativos:

- `--tilt-azimuth-udeg`: fija directamente `mirror_2_azimuth - mirror_1_azimuth`.
- `--tilt-cenital-udeg`: fija directamente `mirror_2_cenital - mirror_1_cenital`.
- Las opciones antiguas con sufijo `urad` siguen disponibles como compatibilidad.

Camara y visualizacion:

- `--pixels-x`, `--pixels-y`: resolucion numerica.
- `--camera-width-mm`, `--camera-height-mm`: campo fisico de vision.
- `--zoom`: zoom numerico solo sobre el campo de vision de la camara.
- `--profile-zoom`: zoom visual del grafico del perfil perpendicular.
- `--profile-samples`: puntos usados en el perfil perpendicular a las franjas.

Haces:

- `--intensity-1`, `--intensity-2`: intensidades relativas.
- `--contrast`: contraste experimental adicional entre 0 y 1.

## Uso rapido

```powershell
python -m michelson.cli --source rectangular --wavelength-nm 632.8 --width-nm 20 --mirror-2-azimuth-udeg 143239 --mirror-2-cenital-udeg 28648 --output-dir outputs/demo
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

- fuente monocromatica, rectangular uniforme o espectro CSV;
- inclinacion azimutal/cenital de ambos espejos;
- desplazamiento axial del espejo 2;
- OPD fija, zoom de camara, zoom de perfil, campo de camara, resolucion, intensidades y contraste.

La vista superior muestra el patron de la camara. La linea cyan indica por
donde se toma el perfil perpendicular a las franjas. La vista inferior muestra
la intensidad, la visibilidad teorica y la visibilidad estimada por
maximos/minimos locales.

## Ejemplos

Fuente monocromatica ideal con inclinacion relativa directa:

```powershell
python -m michelson.cli --source mono --wavelength-nm 632.8 --tilt-azimuth-udeg 5730 --tilt-cenital-udeg 0 --output-dir outputs/mono
```

Ambos espejos inclinados: solo importa la diferencia relativa para las franjas.
Aqui el espejo 1 esta en 17189 u° y el espejo 2 en 160428 u°, por tanto la
inclinacion relativa azimutal equivale a 2500 urad.

```powershell
python -m michelson.cli --mirror-1-azimuth-udeg 17189 --mirror-1-cenital-udeg 5730 --mirror-2-azimuth-udeg 160428 --mirror-2-cenital-udeg 34377 --output-dir outputs/two_mirrors
```

Desplazamiento axial del espejo 2. Moverlo 5 um cambia la OPD central en 10 um:

```powershell
python -m michelson.cli --mirror-2-displacement-um 5 --output-dir outputs/displaced
```

Fuente rectangular de espectro ancho:

```powershell
python -m michelson.cli --source rectangular --wavelength-nm 632.8 --width-nm 20 --opd0-um 0 --tilt-azimuth-udeg 257831 --tilt-cenital-udeg 0 --output-dir outputs/rectangular_20nm
```

Zoom optico de la camara simulada. Este zoom solo recorta la vista de franjas;
el perfil perpendicular sigue usando el ancho fisico configurado de la camara:

```powershell
python -m michelson.cli --zoom 3 --output-dir outputs/zoom_3
```

Zoom visual del perfil perpendicular:

```powershell
python -m michelson.cli --profile-zoom 4 --output-dir outputs/profile_zoom_4
```

## Longitud de coherencia

Para espectros rectangulares, el programa reporta el FWHM aproximado de la
envolvente de visibilidad en OPD:

```text
Lc ~= 1.206709 * lambda0^2 / Delta_lambda
```

Tambien reporta el primer cero aproximado:

```text
L0 ~= lambda0^2 / Delta_lambda
```

El programa compara esos valores con el FWHM obtenido numericamente desde la
visibilidad calculada en el perfil. Si el campo de vision o la inclinacion no
cubren suficiente rango de OPD, el resumen lo indica.

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
