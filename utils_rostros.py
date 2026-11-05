# -*- coding: utf-8 -*-
import json
import os

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'Data')
MODELO_PATH = os.path.join(BASE_DIR, 'modeloLBPHFace.xml')
ETIQUETAS_PATH = os.path.join(BASE_DIR, 'etiquetas.json')

PERSONA = 'Ignacio'
CLASE_DESCONOCIDO = 'Desconocido'
TAM_ROSTRO = (150, 150)
UMBRAL_LBPH = 79.0
MAX_POR_CLASE = 120


def crear_detector():
    ruta = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    detector = cv2.CascadeClassifier(ruta)
    if detector.empty():
        raise RuntimeError('No se pudo cargar el clasificador Haar: ' + ruta)
    return detector


def detectar_rostros(detector, gris, escala=0.5, min_rostro=80):
    if escala != 1.0:
        chico = cv2.resize(gris, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    else:
        chico = gris
    minimo = max(20, int(min_rostro * escala))
    cajas = detector.detectMultiScale(chico, 1.3, 5, minSize=(minimo, minimo))
    if escala == 1.0 or len(cajas) == 0:
        return cajas
    inv = 1.0 / escala
    alto, ancho = gris.shape[:2]
    salida = []
    for (x, y, w, h) in cajas:
        x, y = int(x * inv), int(y * inv)
        w, h = int(w * inv), int(h * inv)
        w = min(w, ancho - x)
        h = min(h, alto - y)
        if w > 0 and h > 0:
            salida.append((x, y, w, h))
    return salida


def preprocesar_rostro(gris, x, y, w, h):
    rostro = gris[y:y + h, x:x + w]
    rostro = cv2.resize(rostro, TAM_ROSTRO, interpolation=cv2.INTER_CUBIC)
    return cv2.equalizeHist(rostro)


def abrir_camara(fuente=0):
    if isinstance(fuente, str) and not fuente.isdigit():
        cap = cv2.VideoCapture(fuente)
    else:
        cap = cv2.VideoCapture(int(fuente), cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError('No se pudo abrir la fuente de video: %r' % (fuente,))
    return cap


def guardar_etiquetas(etiquetas):
    with open(ETIQUETAS_PATH, 'w', encoding='utf-8') as f:
        json.dump(etiquetas, f, ensure_ascii=False, indent=2)


def cargar_etiquetas():
    if not os.path.exists(ETIQUETAS_PATH):
        raise RuntimeError(
            'Falta %s. Entrena primero con: python entrenando-rf.py' % ETIQUETAS_PATH)
    with open(ETIQUETAS_PATH, encoding='utf-8') as f:
        return json.load(f)


def listar_personas():
    if not os.path.isdir(DATA_PATH):
        return []
    return sorted(d for d in os.listdir(DATA_PATH)
                  if os.path.isdir(os.path.join(DATA_PATH, d)))


def _reiluminar(img, gamma=1.0, lado=1.0, invertir=False):
    out = img
    if gamma != 1.0:
        out = (np.power(out / 255.0, gamma) * 255).astype(np.uint8)
    if lado != 1.0:
        grad = np.linspace(lado, 1.0, out.shape[1], dtype=np.float32)
        if invertir:
            grad = grad[::-1]
        out = np.clip(out * grad[None, :], 0, 255).astype(np.uint8)
    return out


VARIANTES_LUZ = (
    dict(),
    dict(gamma=1.5),
    dict(gamma=0.7),
    dict(lado=0.65),
    dict(lado=0.65, invertir=True),
)


def expandir_iluminacion(img):
    return [cv2.equalizeHist(_reiluminar(img, **v)) for v in VARIANTES_LUZ]


def submuestrear(imgs, maximo):
    if maximo <= 0 or len(imgs) <= maximo:
        return list(imgs)
    idx = np.linspace(0, len(imgs) - 1, maximo).round().astype(int)
    return [imgs[i] for i in idx]


class SuavizadorVotos:
    def __init__(self, memoria=7, dist_max=80):
        self.memoria = memoria
        self.dist_max = dist_max
        self.pistas = []

    def votar(self, caja, etiqueta):
        x, y, w, h = caja
        centro = (x + w // 2, y + h // 2)

        mejor, mejor_d = None, self.dist_max
        for pista in self.pistas:
            d = abs(pista['centro'][0] - centro[0]) + abs(pista['centro'][1] - centro[1])
            if d < mejor_d:
                mejor, mejor_d = pista, d

        if mejor is None:
            mejor = {'centro': centro, 'votos': []}
            self.pistas.append(mejor)

        mejor['centro'] = centro
        mejor['votos'].append(etiqueta)
        del mejor['votos'][:-self.memoria]

        votos = mejor['votos']
        return max(set(votos), key=votos.count)

    def limpiar(self, cajas):
        centros = [(x + w // 2, y + h // 2) for (x, y, w, h) in cajas]
        vivas = []
        for pista in self.pistas:
            for cx, cy in centros:
                if abs(pista['centro'][0] - cx) + abs(pista['centro'][1] - cy) < self.dist_max:
                    vivas.append(pista)
                    break
        self.pistas = vivas
