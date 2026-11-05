# -*- coding: utf-8 -*-
import argparse
import os

import cv2

from utils_rostros import (CLASE_DESCONOCIDO, DATA_PATH, PERSONA, abrir_camara,
                           crear_detector, detectar_rostros, preprocesar_rostro)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('nombre', nargs='?', default=PERSONA,
                    help='Nombre de la carpeta a crear dentro de Data/ (def: %s)' % PERSONA)
    ap.add_argument('-n', '--cantidad', type=int, default=300,
                    help='Cuantos rostros guardar (def: 300)')
    ap.add_argument('-f', '--fuente', default='0',
                    help='0 = webcam, o la ruta de un archivo de video')
    ap.add_argument('--salto', type=int, default=1,
                    help='Guardar 1 de cada N rostros detectados (def: 1)')
    args = ap.parse_args()

    person_path = os.path.join(DATA_PATH, args.nombre)
    if not os.path.exists(person_path):
        os.makedirs(person_path)
        print('Carpeta creada:', person_path)
    else:
        print('Agregando rostros a:', person_path)

    existentes = [f for f in os.listdir(person_path) if f.endswith('.jpg')]
    count = len(existentes)
    objetivo = count + args.cantidad

    cap = abrir_camara(args.fuente)
    detector = crear_detector()
    vistos = 0

    print('Capturando... ESC para salir.')
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        alto, ancho = frame.shape[:2]
        if ancho != 640:
            frame = cv2.resize(frame, (640, int(alto * 640 / ancho)))

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostros = detectar_rostros(detector, gris)

        for (x, y, w, h) in rostros:
            vistos += 1
            if vistos % args.salto != 0:
                continue
            rostro = preprocesar_rostro(gris, x, y, w, h)
            cv2.imwrite(os.path.join(person_path, 'rostro_%04d.jpg' % count), rostro)
            count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, '%s: %d/%d' % (args.nombre, count, objetivo),
                    (10, 30), 2, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Capturando rostros (ESC para salir)', frame)

        if cv2.waitKey(1) == 27 or count >= objetivo:
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Listo. %d imagenes en %s' % (count, person_path))
    if args.nombre != CLASE_DESCONOCIDO:
        print('Sugerencia: captura tambien rostros ajenos con:')
        print('    python capturando-rostros.py %s' % CLASE_DESCONOCIDO)


if __name__ == '__main__':
    main()
