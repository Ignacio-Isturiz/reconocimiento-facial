# -*- coding: utf-8 -*-
import argparse
import os

import cv2
import numpy as np

from utils_rostros import (CLASE_DESCONOCIDO, DATA_PATH, ETIQUETAS_PATH,
                           MAX_POR_CLASE, MODELO_PATH, PERSONA,
                           expandir_iluminacion, guardar_etiquetas,
                           listar_personas, submuestrear)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-por-clase', type=int, default=MAX_POR_CLASE,
                    help='Imagenes originales por persona (def: %d); 0 = usar todas.' % MAX_POR_CLASE)
    ap.add_argument('--sin-aug', action='store_true',
                    help='No generar variantes de iluminacion')
    args = ap.parse_args()

    personas = listar_personas()
    if not personas:
        raise SystemExit(
            'No hay nada en %s. Captura rostros primero:\n'
            '    python capturando-rostros.py %s' % (DATA_PATH, PERSONA))

    print('Personas encontradas:', personas)

    labels, facesData = [], []
    for label, nombre in enumerate(personas):
        person_path = os.path.join(DATA_PATH, nombre)
        archivos = sorted(f for f in os.listdir(person_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        imgs = [cv2.imread(os.path.join(person_path, f), 0) for f in archivos]
        imgs = [i for i in imgs if i is not None]
        if not imgs:
            raise SystemExit('La carpeta %s esta vacia.' % person_path)

        usadas = submuestrear(imgs, args.max_por_clase)
        for img in usadas:
            vs = [img] if args.sin_aug else expandir_iluminacion(img)
            facesData += vs
            labels += [label] * len(vs)

        print('  label %d - %-15s %d capturas -> %d usadas -> %d con variantes'
              % (label, nombre, len(imgs), len(usadas),
                 len(usadas) * (1 if args.sin_aug else 5)))

    if PERSONA not in personas:
        print('\nAVISO: no existe la carpeta Data/%s. Revisa el nombre.' % PERSONA)
    if CLASE_DESCONOCIDO not in personas:
        print('\nAVISO: sin clase negativa, el rechazo depende solo del umbral.')
        print('    python capturando-rostros.py %s' % CLASE_DESCONOCIDO)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8)

    print('\nEntrenando con %d imagenes...' % len(facesData))
    if len(facesData) > 2000:
        print('(son muchas: el modelo quedara pesado y bajaran los fps)')
    face_recognizer.train(facesData, np.array(labels))

    face_recognizer.write(MODELO_PATH)
    guardar_etiquetas(personas)
    print('Modelo guardado en   :', MODELO_PATH)
    print('Etiquetas guardadas  :', ETIQUETAS_PATH)
    print('\nAhora corre:  python calibrar-umbral.py')
    print('o directamente:  python reconocimiento-facial.py')


if __name__ == '__main__':
    main()
