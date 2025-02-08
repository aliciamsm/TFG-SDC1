# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 22:23:17 2025

@author: Alicia
"""

#%%imports
import numpy as np
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy import wcs
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from matplotlib.colors import LogNorm
import scipy
from astropy.nddata import Cutout2D
import astropy.units as u
from astropy.coordinates import SkyCoord,match_coordinates_sky, match_coordinates_3d
from time import time 
from astropy.table import Table
from astropy.table import vstack
from astropy.table import hstack
from astropy.coordinates import search_around_sky
from astropy.io import ascii
import pandas as pd
from ska_sdc import Sdc1Scorer
from astropy.wcs import WCS
from astropy.visualization import astropy_mpl_style
from matplotlib import cm

#Se definien las funciones:
#Función que reescribe el catálog para que conserve las entradas incluidas en un array de índices
def cat_match(cat, idx):
    catmatch=Table()
    catmatch['id']=cat['id'][idx]
    catmatch['ra_core']=cat['ra_core'][idx]
    catmatch['dec_core']=cat['dec_core'][idx]
    catmatch['ra_cent']=cat['ra_cent'][idx]
    catmatch['dec_cent']=cat['dec_cent'][idx]
    catmatch['flux']=cat['flux'][idx]
    catmatch['flux']=cat['flux'][idx]
    catmatch['core_frac']=cat['core_frac'][idx]
    catmatch['b_maj']=cat['b_maj'][idx]
    catmatch['b_min']=cat['b_min'][idx]
    catmatch['pa']=cat['pa'][idx]
    catmatch['size']=cat['size'][idx]
    catmatch['class']=cat['class'][idx]
    return catmatch

#Función que añade los objetos que aparecen en 1 banda clasificados al catálogo final
def add_rows_1banda(catb, banda):
    for i in range(banda['flux'].size):
        catb.add_row([banda['id'][i], banda['ra_core'][i], banda['dec_core'][i], banda['ra_cent'][i], banda['dec_cent'][i], banda['flux'][i], banda['core_frac'][i], banda['b_maj'][i], banda['b_min'][i], banda['pa'][i], banda['size'][i], banda['class'][i]])
    return catb

#Función que añade los objetos que aparecen en 2 bandas clasificados al catálogo final
def add_rows_2bandas(catb, banda, clase):
    for i in range(banda['flux'].size):
        catb.add_row([banda['id'][i], banda['ra_core'][i], banda['dec_core'][i], banda['ra_cent'][i], banda['dec_cent'][i], banda['flux'][i], banda['core_frac'][i], banda['b_maj'][i], banda['b_min'][i], banda['pa'][i], banda['size'][i], clase[i]])
    return catb
       
#%%Se abren los catálogos de las 3 bandas
tcatB1 = pd.read_table('cat1_B1.txt', sep='\s+')
tcatB1=tcatB1.dropna()
tcatB2 = pd.read_table('cat1_B2.txt', sep='\s+')
tcatB2=tcatB2.dropna()
tcatB5 = pd.read_table('cat1_B5.txt', sep='\s+')
tcatB5=tcatB5.dropna()

#Primero, se identifican los objetos que aparecen en las 3 bandas
#Se va a emplear el método Sdc1Scorer, que es la forma de correlacionar catálogos para puntuarlos desarrollado por SKAO
#tiene una función que devuelve un dataframe con la información de todos los matches entre dos catálogos, que es lo que se va a aprovechar

#Primero, se determina qué objetos de B5 están en B2
scorer52 = Sdc1Scorer(
    tcatB5,
    tcatB2,
    freq=1400
)
scorer52.run(mode=1, detail=True)
match52=scorer52.score.match_df #df que contiene los objetos comunes a B2 y B5
match52=match52.drop(columns=['a_flux', 'conv_size', 'a_flux_t', 'conv_size_t', 'multi_d_err'])

#Datos de dichos objetos en B5
B5inB2=Table()
B5inB2['id']=match52['id']
B5inB2['ra_core']=match52['ra_core']
B5inB2['dec_core']=match52['dec_core']
B5inB2['ra_cent']=match52['ra_cent']
B5inB2['dec_cent']=match52['dec_cent']
B5inB2['flux']=match52['flux']
B5inB2['core_frac']=match52['core_frac']
B5inB2['b_maj']=match52['b_maj']
B5inB2['b_min']=match52['b_min']
B5inB2['pa']=match52['pa']
B5inB2['size']=match52['size_id']
B5inB2['class']=match52['class']

#Datos de dichos objetos en B2
B2inB5=Table()
B2inB5['id']=match52['id_t']
B2inB5['ra_core']=match52['ra_core_t']
B2inB5['dec_core']=match52['dec_core_t']
B2inB5['ra_cent']=match52['ra_cent_t']
B2inB5['dec_cent']=match52['dec_cent_t']
B2inB5['flux']=match52['flux_t']
B2inB5['core_frac']=match52['core_frac_t']
B2inB5['b_maj']=match52['b_maj_t']
B2inB5['b_min']=match52['b_min_t']
B2inB5['pa']=match52['pa_t']
B2inB5['size']=match52['size_id_t']
B2inB5['class']=match52['class_t']


#Se determina qué objetos de B2 que están en B5 están también en B1
scorer21 = Sdc1Scorer(
    B2inB5.to_pandas(),
    tcatB1,
    freq=560
)
scorer21.run(mode=1, detail=True)
match21=scorer21.score.match_df #df que contiene los objetos presentes en las 3 bandas
match21=match21.drop(columns=['a_flux', 'conv_size', 'a_flux_t', 'conv_size_t', 'multi_d_err'])

#Datos en B2 de los objetos que están en las 3 bandas 
B2inB1=Table()
B2inB1['id']=match21['id']
B2inB1['ra_core']=match21['ra_core']
B2inB1['dec_core']=match21['dec_core']
B2inB1['ra_cent']=match21['ra_cent']
B2inB1['dec_cent']=match21['dec_cent']
B2inB1['flux']=match21['flux']
B2inB1['core_frac']=match21['core_frac']
B2inB1['b_maj']=match21['b_maj']
B2inB1['b_min']=match21['b_min']
B2inB1['pa']=match21['pa']
B2inB1['size']=match21['size_id']
B2inB1['class']=match21['class']

#Datos en B1 de los objetos que están en las 3 bandas
B1inB2=Table()
B1inB2['id']=match21['id_t']
B1inB2['ra_core']=match21['ra_core_t']
B1inB2['dec_core']=match21['dec_core_t']
B1inB2['ra_cent']=match21['ra_cent_t']
B1inB2['dec_cent']=match21['dec_cent_t']
B1inB2['flux']=match21['flux_t']
B1inB2['core_frac']=match21['core_frac_t']
B1inB2['b_maj']=match21['b_maj_t']
B1inB2['b_min']=match21['b_min_t']
B1inB2['pa']=match21['pa_t']
B1inB2['size']=match21['size_id_t']
B1inB2['class']=match21['class_t']

#Queda ver qué objetos de B5 que están en B2 no están en B1 para retirarlos
#Se definen objetos SkyCoord para hacer un match de coordenadas espaciales
ra2=B2inB1['ra_cent']
dec2=B2inB1['dec_cent']
ra5=B5inB2['ra_cent']
dec5=B5inB2['dec_cent']
coords2=SkyCoord(ra2, dec2, unit="deg")
coords5=SkyCoord(ra5, dec5, unit="deg")

#Se determina la posición de los objetos de B2inB1 en B5 obteniéndose un índice
idx25,sep25,d325 = match_coordinates_sky(coords2, coords5, nthneighbor=1)

#Se retiran de B5 los objetos que no están en B1
B5inB2inB1=cat_match(B5inB2, idx25)

#Se tienen los dataframes B1inB2, B2inB1 y B5inB2inB1 con los objetos que aparecen en 3 bandas con los datos de cada banda

#%% Ahora, se van a separar los objetos que aparecen en 2 de las bandas
#Primero se retiran las variables que ya no son necesarias, y se da nombres más concisos a las qe¡ue hay que conservar
B5in3=B5inB2inB1
B2in3=B2inB1
B1in3=B1inB2

del(scorer21, scorer52)
del(B1inB2, B2inB1, B5inB2inB1)
del(coords2, coords5, ra2, dec2, ra5, dec5, d325, sep25, idx25)
del(match21, match52)

#Primero, se comprueba qué objetos de B1 están en B2
scorer12 = Sdc1Scorer(
    tcatB2,
    tcatB1,
    freq=560
)
scorer12.run(mode=1, detail=True)
match12=scorer12.score.match_df #df que contiene los objetos que están en B1 y B2
match12=match12.drop(columns=['a_flux', 'conv_size', 'a_flux_t', 'conv_size_t', 'multi_d_err'])

#Datos en B1 de los objetos que están en B2 y B1
B1inB2=Table()
B1inB2['id']=match12['id_t']
B1inB2['ra_core']=match12['ra_core_t']
B1inB2['dec_core']=match12['dec_core_t']
B1inB2['ra_cent']=match12['ra_cent_t']
B1inB2['dec_cent']=match12['dec_cent_t']
B1inB2['flux']=match12['flux_t']
B1inB2['core_frac']=match12['core_frac_t']
B1inB2['b_maj']=match12['b_maj_t']
B1inB2['b_min']=match12['b_min_t']
B1inB2['pa']=match12['pa_t']
B1inB2['size']=match12['size_id_t']
B1inB2['class']=match12['class_t']

#Datos en B2 de los objetos que están en B1 y B2
B2inB1=Table()
B2inB1['id']=match12['id']
B2inB1['ra_core']=match12['ra_core']
B2inB1['dec_core']=match12['dec_core']
B2inB1['ra_cent']=match12['ra_cent']
B2inB1['dec_cent']=match12['dec_cent']
B2inB1['flux']=match12['flux']
B2inB1['core_frac']=match12['core_frac']
B2inB1['b_maj']=match12['b_maj']
B2inB1['b_min']=match12['b_min']
B2inB1['pa']=match12['pa']
B2inB1['size']=match12['size_id']
B2inB1['class']=match12['class']

#Se comprueba qué objetos de B5 están en B1 (serán muy pocos objetos)
scorer15 = Sdc1Scorer(
    tcatB5,
    tcatB1,
    freq=560
)
scorer15.run(mode=1, detail=True)
match15=scorer15.score.match_df
match15=match15.drop(columns=['a_flux', 'conv_size', 'a_flux_t', 'conv_size_t', 'multi_d_err'])

#Datos en B1 de objetos que están en B5 y B1
B1inB5=Table()
B1inB5['id']=match15['id_t']
B1inB5['ra_core']=match15['ra_core_t']
B1inB5['dec_core']=match15['dec_core_t']
B1inB5['ra_cent']=match15['ra_cent_t']
B1inB5['dec_cent']=match15['dec_cent_t']
B1inB5['flux']=match15['flux_t']
B1inB5['core_frac']=match15['core_frac_t']
B1inB5['b_maj']=match15['b_maj_t']
B1inB5['b_min']=match15['b_min_t']
B1inB5['pa']=match15['pa_t']
B1inB5['size']=match15['size_id_t']
B1inB5['class']=match15['class_t']

#Datos en B5 de objetos que están en B1 y B5
B5inB1=Table()
B5inB1['id']=match15['id']
B5inB1['ra_core']=match15['ra_core']
B5inB1['dec_core']=match15['dec_core']
B5inB1['ra_cent']=match15['ra_cent']
B5inB1['dec_cent']=match15['dec_cent']
B5inB1['flux']=match15['flux']
B5inB1['core_frac']=match15['core_frac']
B5inB1['b_maj']=match15['b_maj']
B5inB1['b_min']=match15['b_min']
B5inB1['pa']=match15['pa']
B5inB1['size']=match15['size_id']
B5inB1['class']=match15['class']

#A estos grupos hay que retirarles los objetos que aparecen en las 3 bandas
#Se crean objetos de SkyCoord y se hace un match de posición para obtener índices que indiquen qué objetos retirar
ra1in3=B1in3['ra_cent']
dec1in3=B1in3['dec_cent']
coords1in3=SkyCoord(ra1in3, dec1in3, unit="deg")

ra2in3=B2in3['ra_cent']
dec2in3=B2in3['dec_cent']
coords2in3=SkyCoord(ra2in3, dec2in3, unit="deg")

ra5in3=B5in3['ra_cent']
dec5in3=B5in3['dec_cent']
coords5in3=SkyCoord(ra5in3, dec5in3, unit="deg")


ra1in2=B1inB2['ra_cent']
dec1in2=B1inB2['dec_cent']
coords1in2=SkyCoord(ra1in2, dec1in2, unit="deg")

ra1in5=B1inB5['ra_cent']
dec1in5=B1inB5['dec_cent']
coords1in5=SkyCoord(ra1in5, dec1in5, unit="deg")

ra2in5=B2inB5['ra_cent']
dec2in5=B2inB5['dec_cent']
coords2in5=SkyCoord(ra2in5, dec2in5, unit="deg")

#Se obtienen los índices que indican qué objetos están 'repetidos' en el grupo de 3 bandas
idx12,sep12,d312 = match_coordinates_sky(coords1in3, coords1in2, nthneighbor=1)
idx15,sep15,d315= match_coordinates_sky(coords1in3, coords1in5, nthneighbor=1)
idx25,sep25,d325 = match_coordinates_sky(coords2in3, coords2in5, nthneighbor=1)

#Se retiran los objetos que aparecen en las 3 bandas de los grupos de 2 bandas
B1inB2.remove_rows(idx12)
B2inB1.remove_rows(idx12)
B1inB5.remove_rows(idx15)
B5inB1.remove_rows(idx15)
B2inB5.remove_rows(idx25)
B5inB2.remove_rows(idx25)

#%%Por último, queda obtener los objetos que aparecen en una sola banda
#Para ello es suficiente con retirar de la banda todos lo objetos que aparecen en las 3 bandas y en 2 de las bandas
#A la banda B1 hay que quitarle B1in3, B1inB2 y B1inB5
#A la banda B2 hay que quitarle B2in3, B1inB2 y B2inB5
#A la banda B5 hay que quitarle B5in3, B1inB5 y B2inB5

#Se definen objetos SkyCoord para cada banda para porde hacer matches de posición
ra1=tcatB1['ra_cent']
dec1=tcatB1['dec_cent']
coords1=SkyCoord(ra1, dec1, unit="deg")

ra2=tcatB2['ra_cent']
dec2=tcatB2['dec_cent']
coords2=SkyCoord(ra2, dec2, unit="deg")

ra5=tcatB5['ra_cent']
dec5=tcatB5['dec_cent']
coords5=SkyCoord(ra5, dec5, unit="deg")

#Se eliminan las variables que ya no son necesarias
del(coords1in2, coords1in5, coords2in5)
del(d312, d315, d325)
del(dec1in2, dec1in5, dec2in5)
del(idx12, idx15, idx25)
del(match12, match15)
del(ra1in2, ra1in5, ra2in5)
del(scorer12, scorer15)
del(sep12, sep15, sep25)

#Se definen las tablas que contendrán los objetos que estén sólo en 1 banda
B1=Table.from_pandas(tcatB1)
B2=Table.from_pandas(tcatB2)
B5=Table.from_pandas(tcatB5)

#Primero, se retiran los objetos que aparecen en las 3 bandas
idx13,sep13,d13=match_coordinates_sky(coords1in3, coords1, nthneighbor=1)
idx23,sep23,d23=match_coordinates_sky(coords2in3, coords2, nthneighbor=1)
idx53,sep53,d53=match_coordinates_sky(coords5in3, coords5, nthneighbor=1)

B1.remove_rows(idx13)
B2.remove_rows(idx23)
B5.remove_rows(idx53)


#Ahora hay que quitar dos grupos más a cada banda, hay que rehacer los objetos SkyCoord porque las tablas B1. B2 y B5 han cambiado de tamaño

#Primero, se retira B1inB2 de B1 y de B2
#Se definen los objetos de coordenadas
ra12=B1inB2['ra_cent']
dec12=B1inB2['dec_cent']
coords12=SkyCoord(ra12, dec12, unit="deg")

ra1=B1['ra_cent']
dec1=B1['dec_cent']
coords1=SkyCoord(ra1, dec1, unit="deg")

ra2=B2['ra_cent']
dec2=B2['dec_cent']
coords2=SkyCoord(ra2, dec2, unit="deg")

#Se hace un match de posición y se retiran los objetos que aparecen en B1 y B2 de las tablas
idx12,sep12,d12=match_coordinates_sky(coords12, coords1, nthneighbor=1)
idx21,sep21,d21=match_coordinates_sky(coords12, coords2, nthneighbor=1)
B1.remove_rows(idx12)
B2.remove_rows(idx21)


#Ahora, se retira B1inB5 de B1 y B5 siguiendo el mismo procedimiento
ra15=B1inB5['ra_cent']
dec15=B1inB5['dec_cent']
coords15=SkyCoord(ra15, dec15, unit="deg")

ra1=B1['ra_cent']
dec1=B1['dec_cent']
coords1=SkyCoord(ra1, dec1, unit="deg")

ra5=B5['ra_cent']
dec5=B5['dec_cent']
coords5=SkyCoord(ra5, dec5, unit="deg")

idx15,sep15,d15=match_coordinates_sky(coords15, coords1, nthneighbor=1)
idx51,sep51,d51=match_coordinates_sky(coords15, coords5, nthneighbor=1)
B1.remove_rows(idx15)
B5.remove_rows(idx51)

#Por último, se eliminan los objetos que aparecen en B2inB5 de B2 y B5
ra25=B2inB5['ra_cent']
dec25=B2inB5['dec_cent']
coords25=SkyCoord(ra25, dec25, unit="deg")

ra2=B2['ra_cent']
dec2=B2['dec_cent']
coords2=SkyCoord(ra2, dec2, unit="deg")

ra5=B5['ra_cent']
dec5=B5['dec_cent']
coords5=SkyCoord(ra5, dec5, unit="deg")

idx25,sep25,d25=match_coordinates_sky(coords25, coords2, nthneighbor=1)
idx52,sep52,d52=match_coordinates_sky(coords25, coords5, nthneighbor=1)
B2.remove_rows(idx25)
B5.remove_rows(idx52)

#Ahora B1, B2 y B5 contienen sólo objetos que no aparecen en el resto de bandas

#Se imprimen los siguientes datos para comprobar que el nñumero de objetos por separado coincide con el tamaño de la banda inicial
#Hacer un match perfeto es muy difícil, una diferencia de unos pocos objetos es aceptable porque representa un porcentaje muy pequeño del catálogo total
catB1size=tcatB1['ra_cent'].size
catB2size=tcatB2['ra_cent'].size
catB5size=tcatB5['ra_cent'].size
sumaB1=B1in3['ra_cent'].size+B1inB2['ra_cent'].size+B1inB5['ra_cent'].size+B1['ra_cent'].size
sumaB2=B2in3['ra_cent'].size+B1inB2['ra_cent'].size+B2inB5['ra_cent'].size+B2['ra_cent'].size
sumaB5=B5in3['ra_cent'].size+B1inB5['ra_cent'].size+B2inB5['ra_cent'].size+B5['ra_cent'].size
print("tamaño B1 original: {}".format(catB1size))
print("tamaño B1 suma: {}".format(sumaB1))
print("diferencia: {}".format(sumaB1-catB1size))

print("tamaño B2 original: {}".format(catB2size))
print("tamaño B2 suma: {}".format(sumaB2))
print("diferencia: {}".format(sumaB2-catB2size))

print("tamaño B5 original: {}".format(catB5size))
print("tamaño B5 suma: {}".format(sumaB5))
print("diferencia: {}".format(sumaB5-catB5size))

#Se eliminan las variables que no son imprescindibles
del(coords1, coords1in3, coords2, coords2in3, coords5, coords5in3, coords12, coords15, coords25)
del(d12, d13, d15, d21, d23, d25, d51, d52, d53)
del(dec1in3, dec2, dec2in3, dec5, dec5in3, dec12, dec15, dec25, dec1)
del(idx12, idx13, idx15, idx21, idx23, idx25, idx51, idx52, idx53)
del(ra1, ra1in3, ra2, ra2in3, ra5, ra5in3, ra12, ra15, ra25)
del(sep12, sep13, sep15, sep21, sep23, sep25, sep51, sep52, sep53)
del(catB1size, catB2size, catB5size)
del(sumaB1, sumaB2, sumaB5)

#%%Una vez separados los objetos en grupos según su aparición en las bandas, queda clasificarlos por su posición en los diagramas de color y flujo
#Se crean unas tablas vacías para irles añadiendo los objetos clasificados con su clase nueva
catb1=Table( names=('id', 'ra_core', 'dec_core', 'ra_cent', 'dec_cent', 'flux', 'core_frac', 'b_maj', 'b_min', 'pa', 'size', 'class'))
catb2=Table( names=('id', 'ra_core', 'dec_core', 'ra_cent', 'dec_cent', 'flux', 'core_frac', 'b_maj', 'b_min', 'pa', 'size', 'class'))
catb5=Table( names=('id', 'ra_core', 'dec_core', 'ra_cent', 'dec_cent', 'flux', 'core_frac', 'b_maj', 'b_min', 'pa', 'size', 'class'))
#Los valores posibles son 1 SS-AGN, 2 FS-AGN y 3 SFG

#Primero se asigna la clase a los objetos que aparecen en una banda
#El criterio establecido es:
B1['class']=1
B2['class']=2
B5['class']=3

#Se les asigna su valor y se añaden al catálogo final de cada banda      
catb1=add_rows_1banda(catb1, B1)
catb2=add_rows_1banda(catb2, B2)
catb5=add_rows_1banda(catb5, B5)
#%%Ahora se asigna la clase a los objetos que aparecen en 2 bandas, a través de las regiones definidas en los diagramas de flujos

#Objetos en B1 y B2
#Los objetos de B1inB2 irán a catb1
#Los objetos de B2inB1 irán a catb2

claseB1B2=np.linspace(0, 0, B1inB2['flux'].size) #array vacío para meter la clase de los objetos, como esán ordenados sirve para las dos bandas

#Este bucle asigna una clase a cada objeto en función de su posción en el diagrama
for i in range(B1inB2['flux'].size):
    if B1inB2['flux'][i]<9e-8:
        claseB1B2[i]=1
    else:
        if B2inB1['flux'][i]/B1inB2['flux'][i]<=0.25:
            claseB1B2[i]=3
        if B2inB1['flux'][i]/B1inB2['flux'][i]>0.25:
            claseB1B2[i]=2

#Figura que muestra cada objeto, su posición en el diagrama y su clase correspondiente
flujo1=B1inB2['flux']
flujo2=B2inB1['flux']
flux=np.linspace(0, 10, flujo1.size)
l1=np.linspace(9e-8, 9e-8, flujo1.size)
X = flujo1
Y = flujo2
fig = plt.figure()
plt.scatter(flujo1, flujo2, c=claseB1B2, cmap='RdYlBu')
plt.plot(flux, flux/4, 'green')
plt.plot(l1, flux)
plt.legend(["class", "x=4y", "x=9e-8"])
plt.xlabel('flujo en B1')
plt.ylabel('flujo en B2')
plt.title('Objetos en bandas B1 y B2 clasificados')
plt.colorbar()
plt.loglog()
plt.show()

#Se guardan en cada catálogo los objetos con su clase correspondiente
catb1=add_rows_2bandas(catb1, B1inB2, claseB1B2)
catb2=add_rows_2bandas(catb2, B2inB1, claseB1B2)

#%%Objetos en B1 y B5
#Se sigue el mismo procedimiento
claseB1B5=np.linspace(0, 0, B1inB5['flux'].size) #array vacío para añadir la clase

#Bucle que asigna una clase a cada objeto segñun su posición del diagrama
for i in range(B1inB5['flux'].size):
    if B1inB5['flux'][i]<8e-8:
        claseB1B5[i]=1
    else:
        if B5inB1['flux'][i]/B1inB5['flux'][i]<=(1/3):
            claseB1B5[i]=3
        if B5inB1['flux'][i]/B1inB5['flux'][i]>(1/3):
            claseB1B5[i]=2

#Figura con los objetos en el diagrama de flujo
flujo1=B1inB5['flux']
flujo2=B5inB1['flux']
flux=np.linspace(0, 10, flujo1.size)
l1=np.linspace(8e-8, 8e-8, flujo1.size)
X = flujo1
Y = flujo2
fig = plt.figure()
plt.scatter(flujo1, flujo2, c=claseB1B5, cmap='RdYlBu')
plt.plot(3*flux, flux)
plt.plot(l1, flux)
plt.legend(["class", "x=3y", "x=8e-8"])
plt.xlabel('flujo en B1')
plt.ylabel('flujo en B5')
plt.title('Objetos en bandas B1 y B5 clasificados')
plt.colorbar()
plt.loglog()
plt.show()

#Se añaden los objetos al catálogo final con su clase
catb1=add_rows_2bandas(catb1, B1inB5, claseB1B5)
catb5=add_rows_2bandas(catb5, B5inB1, claseB1B5)

#%%Objetos en B2 y B5

claseB2B5=np.linspace(0, 0, B2inB5['flux'].size) #array para añadir la clase

#Bucle que asigna la clase:
for i in range(B2inB5['flux'].size):
    if B5inB2['flux'][i]<9e-9:
        claseB2B5[i]=1
    else:
        if B2inB5['flux'][i]/B5inB2['flux'][i]<=1.5:
            claseB2B5[i]=2
        if B2inB5['flux'][i]/B5inB2['flux'][i]>1.5:
            claseB2B5[i]=3
            

#Figura del diagrama de flujo
flujo1=B2inB5['flux']
flujo2=B5inB2['flux']
flux=np.linspace(0, 10, flujo1.size)
l2=np.linspace(9e-9, 9e-9, flujo1.size)
X = flujo1
Y = flujo2
fig = plt.figure()
plt.scatter(flujo1, flujo2, c=claseB2B5, cmap='RdYlBu')
plt.plot(1.5*flux,flux)
plt.plot(flux, l2)
plt.legend(["class", "x=1.5y", "y=9e-9"])
plt.xlabel('flujo en B2')
plt.ylabel('flujo en B5')
plt.title('Objetos en bandas B2 y B5 clasificados')
plt.loglog()
plt.colorbar()
plt.show()

#Se añaden los objetos clasificados al catálogo
catb2=add_rows_2bandas(catb2, B2inB5, claseB2B5)
catb5=add_rows_2bandas(catb5, B5inB2, claseB2B5)


#%%Por último, se clasifican los objetos que aparecen en las 3 bandas

#Se definen los colores del diagrama de color a partir de los flujos en diferentes bandas
color1=B1in3['flux']/B2in3['flux']
color2=B2in3['flux']/B5in3['flux']
claseB1B2B5= np.linspace(0, 0, color1.size) #array vacío que contendrá la clase

#Bucle que asigna la clase en función de la posición en el diagrama de color
for i in range(color1.size):
    if color1[i]<1.8:
        if color2[i]/color1[i]<=3.5:
            claseB1B2B5[i]=1
        if color2[i]/color1[i]>3.5:
            claseB1B2B5[i]=2
    if color1[i]>=1.8:
        if color1[i]<3:
            if color2[i]/color1[i]<=3.5:
                claseB1B2B5[i]=1
            if color2[i]/color1[i]>3.5:
                claseB1B2B5[i]=3
        if color1[i]>=3:
            claseB1B2B5[i]=3

#Diagrama de color con objetos clasificados
X = color1
Y = color2
flux=np.linspace(1, 100, 100)
flux2=np.linspace(6.3, 100000, 100)
l2=np.linspace(1.8 , 1.8, 100)
flux3=np.linspace(0.0001, 10.5, 100)
x3=np.linspace(3, 3, 100)
fig = plt.figure()
plt.scatter(X,Y, c=claseB1B2B5, cmap='RdYlBu')
plt.plot(l2, flux2)
plt.plot(flux, 3.5*flux)
plt.plot(x3, flux3)
plt.loglog()
plt.xlabel('color1(=fluxB1/fluxB2)')
plt.ylabel('color2(=fluxB2/fluxB5)')
plt.legend(["clase", "x=1.8", "x=y/3.5", "x=3"])
plt.title('Objetos en bandas B1, B2 y B5 clasificados')
plt.loglog()
plt.colorbar()
plt.show()

#Se añaden los objetos con su clase al catálogo correspondiente
catb1=add_rows_2bandas(catb1, B1in3, claseB1B2B5)
catb2=add_rows_2bandas(catb2, B2in3, claseB1B2B5)
catb5=add_rows_2bandas(catb5, B5in3, claseB1B2B5)

#%%Finalmente se guardan los catálogos clasificados
catb1.write('cat1_B1_clasificado.txt', format='ascii', overwrite=True)
catb2.write('cat1_B2_clasificado.txt', format='ascii', overwrite=True)
catb5.write('cat1_B5_clasificado.txt', format='ascii', overwrite=True)








