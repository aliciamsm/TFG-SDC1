# TFG-SDC1
Código empleado en el desarrollo del TFG de nombre'Detección, segmentación y clasificación de objetos extragalácticos en los datos del SKA Science Data Challenge 1'
En este repositorio se incluyen los programas escritos para el manejo, caracterización y clasificación de los catálogos propuestos para el SDC1, además del código empleado para modelar el Primary Beam.

El archivo 'modelo_beam_coseno.py' incluye el código para modelar el Primary Beam según una función cosenoidal.
El archivo 'caracterizar_catalogos.py' incluye lo necesario para el manejo de los dos catálogos de umbrales diferentes que se han empleado para armar el catálogo final en cada una de las bandas. Sirve para ordenar las entradas de ambos catálogos según un match posicional, y hacer los cambios necesarios para obtener todas las categorías del catálogo final, menos la clase.
El archivo 'clasificar_catálogos.py' contiene el código empleado para clasificar los diferentes objetos por clases a partir de su presencia en las diferentes bandas y su posición en los diagramas de flujo y color.

Los tres archivos hacen uso de las librerías 'numpy', 'matplotlib' y 'astropy' de Python, además de la librería desarrollada por SKAO para la evaluación del SDC1, cuya documentación se puede consultar en 'https://developer.skatelescope.org/projects/sdc1-scoring/en/latest/index.html'.
