# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 21:39:34 2025

@author: Alicia
"""

#imports
import numpy as np
from astropy.io import fits
from astropy import wcs
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord,match_coordinates_sky, match_coordinates_3d
from astropy.table import Table
from astropy.table import vstack
from astropy.table import hstack
from astropy.coordinates import search_around_sky
from astropy.io import ascii

#Se definien las siguientes funciones
#Función que cambia el origen de coordenadas de la Ascensión Recta (AR) para que estén contenida en [-180, 180], el criterio del true catalogue
def coord_ra(columna):
    for i in range(columna.size):#esta probablemente no va a servir
        if columna[i]>180:
            columna[i]=columna[i]-360
        else:
            columna[i]=columna[i]
    return columna

#Función que adjudica un valor de 0 al core_frac de todos los objetos que tengan un flujo total de 0, para evitar que una división entre 0 registre como NaN a objetos que tienen un core_frac nulo
def core_frac_bueno(flujoB, core_frac):
    corefracbueno=np.linspace(0, 0, flujoB.size)
    for i in range (flujoB.size):
        if flujoB[i]==0:
            corefracbueno[i] = 0
        else: 
            corefracbueno[i]= core_frac[i]
    return corefracbueno

#Función para cambio de unidades de píxel a grados
def pixel_to_deg(ra_pix, pix_size):
    ra_deg=ra_pix*(pix_size/3600)
    return ra_deg

#Función que abre los dos catálogos con umbrales diferentes (A y B a partir de ahora), y cambia el nombre de algunas categorías para un manejo más cómodo
def get_catalogos():
    catA = Table.read('B1_10_7.fits') #ejemplo con el catálogo de umbral 7 para la banda B1 (catálogo 1)
    catA.rename_column('NUMBER','NUMBER_1')
    catA.rename_column('ID_PARENT','ID_PARENT_1')
    catA.rename_column('ALPHAWIN_SKY','RA_1')
    catA.rename_column('DELTAWIN_SKY','DEC_1')
    catA.rename_column('ISOAREA_IMAGE','ISOAREA_IMAGE_1')
    
    catB = Table.read('B1_10.fits') #ejemplo con el catálogo de umbral 10 para la banda B1 (catálogo 1)
    catB.rename_column('NUMBER','NUMBER_2')
    catB.rename_column('ID_PARENT','ID_PARENT_2')
    catB.rename_column('ALPHAWIN_SKY','RA_2')
    catB.rename_column('DELTAWIN_SKY','DEC_2')
    catB.rename_column('ISOAREA_IMAGE', 'ISOAREA_IMAGE_2')
    catB.rename_column('XWIN_IMAGE', 'XWIN_IMAGE_2')
    catB.rename_column('YWIN_IMAGE', 'YWIN_IMAGE_2')
    return catA,catB

#Función que ordena las entradas del catálogo A en función de las coordenadas del catálogo B a partir de un índice idx1
def catA_match_psf(catA, idx1):
    catA_match=Table()
    catA_match['NUMBER_1']=catA['NUMBER_1'][idx1]
    catA_match['ID_PARENT_1']=catA['ID_PARENT_1'][idx1]
    catA_match['RA_1']=catA['RA_1'][idx1]
    catA_match['DEC_1']=catA['DEC_1'][idx1]
    catA_match['ISOAREA_IMAGE_1']=catA['ISOAREA_IMAGE_1'][idx1]
    catA_match['FLUX_PSF']=catA['FLUX_PSF'][idx1]
    catA_match['SPHEROID_REFF_WORLD']=catA['SPHEROID_REFF_WORLD'][idx1]
    catA_match['SPHEROID_ASPECT_IMAGE']=catA['SPHEROID_ASPECT_IMAGE'][idx1]
    catA_match['SPHEROID_THETA_SKY']=catA['SPHEROID_THETA_SKY'][idx1]
    catA_match['VECTOR_ASSOC']=catA['VECTOR_ASSOC'][idx1]
    return catA_match

#Datos: tamaño de píxel en cada banda en arcsec
pix1=0.6
pix2=0.24
pix5=0.037

#Primero se abren los dos catálogos
#Este ejemplo elabora el catálogo 1 para la banda B1
catA,catB=get_catalogos()

#Como se han obtenido en modo ASSOC, ambos catálogos contienen los mismos objetos pero están en desorden
#Se hace un match de coordenadas de posición entre los dos catálogos a partir de las entradas 'VECTOR_ASSOC' en el catálogo A y 'XWIN_IMAGE_2' en el catálogo B

#Se definen coordenadas angulares para cada catálogo
ra1=pixel_to_deg(catA['VECTOR_ASSOC'].data, 0.06)
ra2=pixel_to_deg(catB['XWIN_IMAGE_2'].data, 0.06)
dec1=ra1
dec2=ra2

#Se crea un objeto SkyCoord para cada catálogo, necesario para el match basado en la posición
coords_A = SkyCoord(ra1*u.deg, dec1*u.deg, unit="deg") 
coords_B = SkyCoord(ra2*u.deg, dec2*u.deg, unit="deg") 

#Utilizando el método match_coordinates_sky se obtiene un índice que relaciona las entradas de los dos catálogos
idx1,sep,d3 = match_coordinates_sky(coords_B, coords_A, nthneighbor=1)

#Se reordenan las entradas del catálogo A para que encajen con el catálogo B
catA=catA_match_psf(catA, idx1)
del(coords_A, coords_B, d3, dec1, dec2, idx1, ra1, ra2, sep) #estas variables ya no son necesarias

#Ahora que se tienen los catálogos con las distintas entradas ordenadas, se pueden obtener las categorías del catálogo final
#Cambios de unidades y operaciones para obtener bmaj y bmin (categorías 8 y 9)
bmaj=(catA['SPHEROID_REFF_WORLD']*3600)/(np.sqrt(catA['SPHEROID_ASPECT_IMAGE']))
bmin=bmaj*catA['SPHEROID_ASPECT_IMAGE']

#Cambios de unidades para obtener el flujo de core y centroide en Jy
flujoA=catA['FLUX_PSF']*catA['ISOAREA_IMAGE_1']*((pix1*2*np.pi)/3600)**2 #flujo del objeto completo, categoría 6
flujoB=catB['FLUX_PSF']*catB['ISOAREA_IMAGE_2']*((pix1*2*np.pi)/3600)**2 #flujo del core, es necesario para obtener core_frac

#Cambios de unidades para obtener las entradas relacionadas con la posición (categorias 2, 3,4 y 5)
ra_cent=coord_ra(catA['RA_1'])
ra_core=coord_ra(catB['RA_2'])
dec_cent=catA['DEC_1']
dec_core=catB['DEC_2']

#Se obtiene el core_frac
core_frac=flujoB/flujoA

#Se define una tabla nueva para añadirle las categoráis finales en sus unidades correctas
tabla_nueva = Table()
tabla_nueva['id']=catA['ID_PARENT_1']
tabla_nueva['ra_core']=ra_core
tabla_nueva['dec_core']=dec_core
tabla_nueva['ra_cent']=ra_cent
tabla_nueva['dec_cent']=dec_cent
tabla_nueva['flux']=flujoA
tabla_nueva['core_frac']=core_frac_bueno(flujoB, core_frac)
tabla_nueva['b_maj']=bmaj 
tabla_nueva['b_min']=bmin
tabla_nueva['pa']= catA['SPHEROID_THETA_SKY']
tabla_nueva['size'] = 3 #como corresponde al ajuste a un perfil de Sérsic
tabla_nueva['class'] = 1  #la clasificación de objetos se hace aparte, provisionalmente se asigna clase 1 a todos los objetos para que el catálogo tenga la forma necesaria para manejarlo

#Se guarda el catálogo caracterizado en un archivo de texto
#Las fuentes de este catálogo están sin clasificar
tabla_nueva.write('cat1_B1.txt', format='ascii', overwrite=True) 