# -*- coding: utf-8 -*-
import argparse

import cv2

from utils_rostros import (CLASE_DESCONOCIDO, MODELO_PATH, PERSONA,
                           UMBRAL_LBPH, SuavizadorVotos, abrir_camara,
                           cargar_etiquetas, crear_detector,
                           detectar_rostros, preprocesar_rostro)

VERDE = (0, 255, 0)
ROJO = (0, 0, 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-u', '--umbral', type=float, default=UMBRAL_LBPH,
                    help='Distancia LBPH maxima para aceptar a %s (def: %.0f).' % (PERSONA, UMBRAL_LBPH))
    ap.add_argument('-f', '--fuente', default='0',
                    help='0 = webcam, o la ruta de un archivo de video')
    ap.add_argument('--debug', action='store_true',
                    help='Mostrar la distancia LBPH en pantalla')
    ap.add_argument('--sin-suavizado', action='store_true',
                    help='Decidir frame a frame, sin voto mayoritario')
    args = ap.parse_args()

    etiquetas = cargar_etiquetas()
    print('Etiquetas del modelo:', etiquetas)
    print('Umbral:', args.umbral)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        face_recognizer.read(MODELO_PATH)
    except cv2.error:
        raise SystemExit('No se pudo leer %s. Entrena con: python entrenando-rf.py' % MODELO_PATH)

    detector = crear_detector()
    cap = abrir_camara(args.fuente)
    suavizador = SuavizadorVotos(memoria=7)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostros = detectar_rostros(detector, gris)

        for (x, y, w, h) in rostros:
            rostro = preprocesar_rostro(gris, x, y, w, h)
            label, distancia = face_recognizer.predict(rostro)
            nombre_modelo = etiquetas[label] if label < len(etiquetas) else CLASE_DESCONOCIDO

            if nombre_modelo == PERSONA and distancia < args.umbral:
                nombre = PERSONA
            else:
                nombre = 'DESCONOCIDO'

            if not args.sin_suavizado:
                nombre = suavizador.votar((x, y, w, h), nombre)

            color = VERDE if nombre == PERSONA else ROJO
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, nombre, (x, y - 10), 2, 0.9, color, 2, cv2.LINE_AA)
            if args.debug:
                cv2.putText(frame, '%s %.1f' % (nombre_modelo, distancia),
                            (x, y + h + 22), 1, 1.2, (255, 255, 0), 1, cv2.LINE_AA)

        if not args.sin_suavizado:
            suavizador.limpiar(rostros)

        cv2.imshow('Reconocimiento facial (ESC para salir)', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
