# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 21:32:39 2025

@author: Alicia
"""
#Se importan los siguientes paquetes y librerías
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt


#Se definen las siguientes funciones:
#Función para cambio de unidades de píxel a grados
def pixel_to_deg(ra_pix, pix_size):
    ra_deg=ra_pix*(pix_size/3600)
    return ra_deg

#Función de ajuste a beam cosenoidal bidimensional dependiente del FWHM
def beam_coseno(fwhm, theta):
    x=(theta*1.2)/fwhm
    beam=np.cos(np.pi*x)/(1-4*(x**2))
    return (np.abs(beam))**2

#datos para las distintas bandas: FWHM en grados y tamaño de píxel en arcsec
fwhm1=0.5547
fwhm2=0.2219
fwhm5=0.0342
pix1=0.6
pix2=0.24
pix5=0.037

#Se genera un beam del tamaño de la imagen final. Se emplean los datos correspondientes a B1 como ejemplo (pix1, fwhm1)
x=np.linspace(-16384, 16384, 32768) #array del tamaño de la imagen centrado en el origen
#Se convierte el array a unidades angulares
thetax=pixel_to_deg(x, pix1) 
thetay=thetax

#Se genera una red bidimensional para adjudicarle el valor del beam en cada punto
thetax, thetay = np.meshgrid(thetax, thetay) 
R = np.sqrt(thetax**2 + thetay**2) 
Z = beam_coseno(fwhm5, R) #se genera el beam cosenoidal

#Se abre la imagen a corregir, en est ecaso la de la banda B1
b=fits.open('SKAMid_B1_1000h_v3.fits')
bdata=np.squeeze(b[0].data) #se comprime la imagen en un array bidimensional

#Se abre también el Primary Beam original
pb=fits.open('PrimaryBeam_B5.fits')
pbdata=np.squeeze(pb[0].data)
pbdatanonan = pbdata[~np.isnan(pbdata)]
minimo=pbdatanonan.min() #se obtiene el valor mínimo que toma el beam

#Se corrige la imagen
taperorigen=Z+minimo# hay que desplazar el origen para que el valor mínimo del beam simulado coincida con el del beam original
correccionorigen=bdata/(taperorigen/taperorigen.max())#se divide la imagen del cielo entre el beam normalizado

#Se guarda la imagen corregida, conservando su header original
fits.writeto('B1_corregido_coseno.fits', correccionorigen, b[0].header)