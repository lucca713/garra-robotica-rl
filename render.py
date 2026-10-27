"""
Desenho da cena em OpenGL (pipeline clássico, com glPushMatrix/glPopMatrix).

O OpenGL não participa do aprendizado: ele só lê os números da simulação
(ângulos, abertura da garra, posição da caixa) e desenha.

O braço é feito com MODELAGEM HIERÁRQUICA: cada segmento é desenhado depois
de girar pelo ângulo da sua junta e andar até a ponta do segmento anterior.
Assim, girar o ombro leva junto o cotovelo, o punho e a garra.

Câmera: setas giram a visão, Z e X aproximam/afastam.
"""
import math

import numpy as np
import glfw
from OpenGL.GL import *   # noqa: F401,F403
from OpenGL.GLU import *  # noqa: F401,F403

from ambiente import ALTURA_BASE, ALTURA_ALVO, COMPRIMENTOS, MEIO_LADO_CAIXA

# Cores (R, G, B)
COR_CHAO = (0.55, 0.57, 0.60)
COR_BASE = (0.25, 0.25, 0.30)
COR_SEGMENTO = (0.95, 0.60, 0.10)
COR_JUNTA = (0.20, 0.20, 0.25)
COR_GARRA = (0.70, 0.72, 0.75)
COR_CAIXA = (0.20, 0.45, 0.85)
COR_CAIXA_PRESA = (0.15, 0.75, 0.35)
COR_ALVO = (0.10, 0.80, 0.30)

# Faces de um cubo de lado 1 centrado na origem: (normal, 4 vértices)
_FACES_CUBO = [
    ((0, 0, 1), [(-.5, -.5, .5), (.5, -.5, .5), (.5, .5, .5), (-.5, .5, .5)]),
    ((0, 0, -1), [(-.5, -.5, -.5), (-.5, .5, -.5), (.5, .5, -.5), (.5, -.5, -.5)]),
    ((1, 0, 0), [(.5, -.5, -.5), (.5, .5, -.5), (.5, .5, .5), (.5, -.5, .5)]),
    ((-1, 0, 0), [(-.5, -.5, -.5), (-.5, -.5, .5), (-.5, .5, .5), (-.5, .5, -.5)]),
    ((0, 1, 0), [(-.5, .5, -.5), (-.5, .5, .5), (.5, .5, .5), (.5, .5, -.5)]),
    ((0, -1, 0), [(-.5, -.5, -.5), (.5, -.5, -.5), (.5, -.5, .5), (-.5, -.5, .5)]),
]


def desenhar_cubo():
    glBegin(GL_QUADS)
    for normal, vertices in _FACES_CUBO:
        glNormal3f(*normal)
        for v in vertices:
            glVertex3f(*v)
    glEnd()


def caixa(cor, centro, tamanho):
    """Desenha um paralelepípedo com o centro e o tamanho (x, y, z) dados,
    no sistema de coordenadas atual."""
    glPushMatrix()
    glTranslatef(*centro)
    glScalef(*tamanho)
    glColor3f(*cor)
    desenhar_cubo()
    glPopMatrix()


class Renderizador:
    def __init__(self, largura=960, altura=680, titulo="Garra robotica - Aprendizado por Reforco", visivel=True):
        if not glfw.init():
            raise RuntimeError("Não foi possível iniciar o GLFW.")
        if not visivel:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.janela = glfw.create_window(largura, altura, titulo, None, None)
        if not self.janela:
            glfw.terminate()
            raise RuntimeError("Não foi possível criar a janela OpenGL.")
        glfw.make_context_current(self.janela)
        glfw.swap_interval(1)

        self.aberta = True
        self.cam_azimute = -25.0   # graus, gira em volta do eixo Y
        self.cam_elevacao = 20.0   # graus, acima do chão
        self.cam_distancia = 2.3
        self.quadrica = gluNewQuadric()
        self._configurar_opengl()

    def _configurar_opengl(self):
        glClearColor(0.93, 0.95, 0.98, 1.0)
        glEnable(GL_DEPTH_TEST)          # Z-buffer: o que está na frente cobre o de trás
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.35, 0.35, 0.35, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.85, 0.85, 0.85, 1.0))
        glEnable(GL_COLOR_MATERIAL)      # glColor vira a cor do material
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_NORMALIZE)           # corrige as normais depois do glScale
        glShadeModel(GL_SMOOTH)

    # --------------------------------------------------------------
    # API usada pelo ambiente
    # --------------------------------------------------------------
    def desenhar(self, estado):
        """Desenha um quadro. Devolve False se a janela foi fechada."""
        if not self.aberta or glfw.window_should_close(self.janela):
            self.aberta = False
            return False
        self._desenhar_cena(estado)
        glfw.swap_buffers(self.janela)
        glfw.poll_events()
        return True

    def capturar(self, estado):
        """Desenha e devolve a imagem como array (altura, largura, 3), útil para vídeos."""
        self._desenhar_cena(estado)
        largura, altura = glfw.get_framebuffer_size(self.janela)
        dados = glReadPixels(0, 0, largura, altura, GL_RGB, GL_UNSIGNED_BYTE)
        imagem = np.frombuffer(dados, dtype=np.uint8).reshape(altura, largura, 3)
        glfw.swap_buffers(self.janela)
        glfw.poll_events()
        return np.flipud(imagem).copy()

    def tecla_pressionada(self, tecla):
        return glfw.get_key(self.janela, tecla) == glfw.PRESS

    def fechar(self):
        if self.janela:
            glfw.destroy_window(self.janela)
            self.janela = None
        glfw.terminate()
        self.aberta = False

    # --------------------------------------------------------------
    # Desenho
    # --------------------------------------------------------------
    def _desenhar_cena(self, estado):
        self._mover_camera_pelo_teclado()
        largura, altura = glfw.get_framebuffer_size(self.janela)
        glViewport(0, 0, largura, altura)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Projeção em perspectiva
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, largura / max(altura, 1), 0.05, 50.0)

        # Câmera orbitando um ponto em frente ao braço
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        alvo = (0.35, 0.35, 0.0)
        az = math.radians(self.cam_azimute)
        el = math.radians(self.cam_elevacao)
        olho = (
            alvo[0] + self.cam_distancia * math.cos(el) * math.sin(az),
            alvo[1] + self.cam_distancia * math.sin(el),
            alvo[2] + self.cam_distancia * math.cos(el) * math.cos(az),
        )
        gluLookAt(*olho, *alvo, 0.0, 1.0, 0.0)
        glLightfv(GL_LIGHT0, GL_POSITION, (1.0, 3.0, 2.0, 1.0))

        self._desenhar_chao()
        self._desenhar_linha_alvo()
        self._desenhar_caixa(estado["caixa"], estado["segurando"])
        self._desenhar_braco(estado["angulos"], estado["garra"])

    def _desenhar_chao(self):
        caixa(COR_CHAO, (0.3, -0.01, 0.0), (3.0, 0.02, 1.6))
        glDisable(GL_LIGHTING)
        glColor3f(0.45, 0.47, 0.50)
        glBegin(GL_LINES)
        for i in range(-12, 19):
            x = i * 0.1
            glVertex3f(x, 0.001, -0.8)
            glVertex3f(x, 0.001, 0.8)
        for k in range(-8, 9):
            z = k * 0.1
            glVertex3f(-1.2, 0.001, z)
            glVertex3f(1.8, 0.001, z)
        glEnd()
        glEnable(GL_LIGHTING)

    def _desenhar_linha_alvo(self):
        """Linha verde: altura que a caixa precisa alcançar."""
        y = ALTURA_ALVO + MEIO_LADO_CAIXA
        glDisable(GL_LIGHTING)
        glColor3f(*COR_ALVO)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glVertex3f(0.2, y, 0.0)
        glVertex3f(1.0, y, 0.0)
        glEnd()
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    def _desenhar_caixa(self, centro, segurando):
        lado = 2 * MEIO_LADO_CAIXA
        cor = COR_CAIXA_PRESA if segurando else COR_CAIXA
        caixa(cor, (centro[0], centro[1], 0.0), (lado, lado, lado))

    def _desenhar_braco(self, angulos, garra):
        glPushMatrix()
        # Pedestal
        caixa(COR_BASE, (0.0, ALTURA_BASE / 2, 0.0), (0.22, ALTURA_BASE, 0.22))
        glTranslatef(0.0, ALTURA_BASE, 0.0)

        ultimo = len(COMPRIMENTOS) - 1
        for i, (theta, comprimento) in enumerate(zip(angulos, COMPRIMENTOS)):
            glRotatef(math.degrees(theta), 0.0, 0.0, 1.0)   # gira a junta (eixo Z)
            glColor3f(*COR_JUNTA)
            gluSphere(self.quadrica, 0.045, 20, 20)          # a junta
            # O último segmento é desenhado mais curto para dar espaço à garra.
            visivel = comprimento - 0.06 if i == ultimo else comprimento
            caixa(COR_SEGMENTO, (visivel / 2, 0.0, 0.0), (visivel, 0.05, 0.05))
            glTranslatef(comprimento, 0.0, 0.0)              # vai para a ponta

        self._desenhar_garra(garra)
        glPopMatrix()

    def _desenhar_garra(self, garra):
        """Desenhada com a ponta do braço na origem. Os dedos abrem no eixo Y local."""
        abertura = MEIO_LADO_CAIXA + 0.01 + 0.05 * garra   # distância de cada dedo ao centro
        caixa(COR_GARRA, (-0.065, 0.0, 0.0), (0.02, 2 * (MEIO_LADO_CAIXA + 0.07), 0.07))  # palma
        for lado in (-1, 1):
            caixa(COR_GARRA, (-0.01, lado * abertura, 0.0), (0.10, 0.015, 0.06))       # dedos

    def _mover_camera_pelo_teclado(self):
        if self.tecla_pressionada(glfw.KEY_LEFT):
            self.cam_azimute -= 1.5
        if self.tecla_pressionada(glfw.KEY_RIGHT):
            self.cam_azimute += 1.5
        if self.tecla_pressionada(glfw.KEY_UP):
            self.cam_elevacao = min(self.cam_elevacao + 1.0, 85.0)
        if self.tecla_pressionada(glfw.KEY_DOWN):
            self.cam_elevacao = max(self.cam_elevacao - 1.0, -5.0)
        if self.tecla_pressionada(glfw.KEY_Z):
            self.cam_distancia = max(self.cam_distancia - 0.03, 0.8)
        if self.tecla_pressionada(glfw.KEY_X):
            self.cam_distancia = min(self.cam_distancia + 0.03, 5.0)
