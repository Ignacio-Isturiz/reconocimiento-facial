# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np

from utils_rostros import (CLASE_DESCONOCIDO, DATA_PATH, MAX_POR_CLASE,
                           PERSONA, expandir_iluminacion, listar_personas,
                           submuestrear)


def cargar(nombre):
    carpeta = os.path.join(DATA_PATH, nombre)
    archivos = sorted(f for f in os.listdir(carpeta)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    imgs = [cv2.imread(os.path.join(carpeta, f), 0) for f in archivos]
    return [i for i in imgs if i is not None]


def percentiles(valores, etiqueta):
    if not valores:
        print('  %-14s (ninguno llego a compararse: todos rechazados por label)' % etiqueta)
        return
    a = np.array(valores)
    print('  %-14s n=%-4d min=%.1f  p50=%.1f  p90=%.1f  p95=%.1f  max=%.1f'
          % (etiqueta, len(a), a.min(), np.percentile(a, 50),
             np.percentile(a, 90), np.percentile(a, 95), a.max()))


def main():
    personas = listar_personas()
    if PERSONA not in personas:
        raise SystemExit('No existe Data/%s. Captura rostros primero.' % PERSONA)

    train_imgs, train_labels = [], []
    test = {}
    for label, nombre in enumerate(personas):
        imgs = cargar(nombre)
        if len(imgs) < 10:
            raise SystemExit('Muy pocas imagenes en Data/%s (%d).' % (nombre, len(imgs)))
        corte = int(len(imgs) * 0.8)
        usadas = submuestrear(imgs[:corte], MAX_POR_CLASE)
        for img in usadas:
            vs = expandir_iluminacion(img)
            train_imgs += vs
            train_labels += [label] * len(vs)
        test[nombre] = imgs[corte:]
        print('%-15s entrena=%d (%d con variantes)  prueba=%d'
              % (nombre, len(usadas), len(usadas) * 5, len(imgs) - corte))

    rec = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    print('\nEntrenando modelo de prueba...')
    rec.train(train_imgs, np.array(train_labels))

    label_persona = personas.index(PERSONA)

    def distancia_a_persona(img):
        label, dist = rec.predict(img)
        return dist if label == label_persona else 1e6

    propias = [distancia_a_persona(i) for i in test[PERSONA]]
    ajenas = []
    for nombre, imgs in test.items():
        if nombre != PERSONA:
            ajenas += [distancia_a_persona(i) for i in imgs]

    print('\nDistancias LBPH sobre fotos no vistas (menor = mas parecido a %s):' % PERSONA)
    percentiles([d for d in propias if d < 1e5], PERSONA)
    if ajenas:
        percentiles([d for d in ajenas if d < 1e5], 'otros')
        rechazo_label = sum(1 for d in ajenas if d >= 1e5)
        print('  (%d/%d rostros ajenos ya quedan descartados por label)' % (rechazo_label, len(ajenas)))

    if not ajenas:
        sugerido = float(np.percentile(propias, 95)) + 5
        print('\nNo hay carpeta Data/%s, solo puedo mirar tus propias fotos.' % CLASE_DESCONOCIDO)
        print('Umbral sugerido (p95 + margen): %.0f' % sugerido)
        print('Para una calibracion real:  python capturando-rostros.py %s' % CLASE_DESCONOCIDO)
    else:
        filas = []
        for u in range(20, 131):
            tpr = sum(1 for d in propias if d < u) / len(propias)
            tnr = sum(1 for d in ajenas if d >= u) / len(ajenas)
            filas.append((u, tpr, tnr, (tpr + tnr) / 2))

        mejor_bal = max(f[3] for f in filas)
        empates = [f[0] for f in filas if f[3] == mejor_bal]
        recomendado = min(max(empates), int(round(min(empates) * 1.1)))

        print('\n umbral   acierta %-10s   rechaza otros   balance' % PERSONA)
        for u, tpr, tnr, bal in filas:
            if u % 10 == 0 or u == recomendado:
                marca = '  <== recomendado' if u == recomendado else ''
                print('   %3d      %6.1f%%            %6.1f%%       %5.1f%%%s'
                      % (u, tpr * 100, tnr * 100, bal * 100, marca))
        print('\nUmbral recomendado: %d  (rango igual de bueno: %d-%d)'
              % (recomendado, min(empates), max(empates)))
        print('Usalo asi:  python reconocimiento-facial.py -u %d' % recomendado)
        print('(o cambia UMBRAL_LBPH en utils_rostros.py para dejarlo fijo)')


if __name__ == '__main__':
    main()
