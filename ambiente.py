"""
Ambiente de simulação do braço robótico com garra (padrão Gymnasium).

O braço fica em um plano vertical (eixos X e Y): todas as juntas giram em
torno do MESMO eixo (Z), e cada junta tem ângulo mínimo e máximo, ou seja,
nenhuma dá a volta completa de 360 graus.

O agente de RL (PPO ou SAC) conversa com este arquivo por dois métodos:
    reset()      -> começa um episódio novo e devolve a observação inicial
    step(acao)   -> aplica a ação e devolve (observacao, recompensa,
                    terminou, truncou, info)

O desenho em OpenGL fica em render.py. Esta parte é só matemática, por isso
o treino pode rodar sem abrir janela (bem mais rápido).
"""
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ------------------------------------------------------------------
# Parâmetros do braço (unidades em metros e radianos)
# ------------------------------------------------------------------
ALTURA_BASE = 0.10                            # altura do pedestal
COMPRIMENTOS = np.array([0.50, 0.40, 0.20])   # ombro, cotovelo, punho
# Ângulos RELATIVOS ao segmento anterior. O ombro é medido a partir do chão.
LIMITES_MIN = np.radians([10.0, -150.0, -120.0])
LIMITES_MAX = np.radians([170.0, 150.0, 120.0])
ANGULOS_INICIAIS = np.radians([90.0, -60.0, -60.0])

PASSO_ANGULO_MAX = np.radians(4.0)  # quanto uma junta gira por passo com ação = 1
PASSO_GARRA_MAX = 0.2               # quanto a garra abre/fecha por passo

# ------------------------------------------------------------------
# Parâmetros da caixa e da tarefa
# ------------------------------------------------------------------
MEIO_LADO_CAIXA = 0.04              # caixa de 8 cm de lado
FAIXA_X_CAIXA = (0.35, 0.85)        # onde a caixa pode aparecer (alcançável)
ALCANCE_PEGAR = 0.05                # distância máxima ponta-caixa para agarrar
GARRA_FECHADA = 0.3                 # abertura abaixo disso = garra fechada
GARRA_SOLTA = 0.5                   # abertura acima disso = solta a caixa
ALTURA_ALVO = 0.40                  # quanto levantar a caixa para ter sucesso
QUEDA_POR_PASSO = 0.02              # gravidade simplificada da caixa solta
MAX_PASSOS = 250                    # duração máxima de um episódio


def cinematica_direta(angulos):
    """Devolve a posição (x, y) da base e de cada junta, terminando na ponta.

    Como as juntas estão no mesmo eixo, basta somar os ângulos:
        phi_i = theta_1 + ... + theta_i
        x += L_i * cos(phi_i),  y += L_i * sin(phi_i)
    """
    pontos = [np.array([0.0, ALTURA_BASE])]
    phi = 0.0
    for theta, comprimento in zip(angulos, COMPRIMENTOS):
        phi += theta
        pontos.append(pontos[-1] + comprimento * np.array([np.cos(phi), np.sin(phi)]))
    return np.array(pontos)


class BracoRoboticoEnv(gym.Env):
    """Braço de 3 juntas + garra que precisa pegar e levantar uma caixa."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, recompensa="etapas"):
        """recompensa="etapas": recompensa guiada (padrão, a que funciona).
        recompensa="esparsa": só +10 no sucesso e 0 no resto, para comparação."""
        super().__init__()
        self.render_mode = render_mode
        self.tipo_recompensa = recompensa

        # Ação contínua: 3 juntas + garra, cada valor entre -1 e 1.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observação: 17 números (ver _observacao).
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)

        self._tela = None  # janela OpenGL, criada só quando for desenhar
        self._ultimo_quadro = 0.0

    # --------------------------------------------------------------
    # Interface Gymnasium
    # --------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.angulos = ANGULOS_INICIAIS.copy()
        self.garra = 1.0            # 1 = totalmente aberta, 0 = fechada
        self.segurando = False
        self.ja_pegou = False       # o bônus de pegar só vale uma vez
        self.passos = 0
        x = self.np_random.uniform(*FAIXA_X_CAIXA)
        self.caixa = np.array([x, MEIO_LADO_CAIXA])  # centro da caixa, no chão

        if self.render_mode == "human":
            self.render()
        return self._observacao(), {}

    def step(self, acao):
        acao = np.clip(np.asarray(acao, dtype=np.float64), -1.0, 1.0)
        self.passos += 1

        # 1) Aplica a ação respeitando os limites das juntas (sem 360 graus).
        self.angulos = np.clip(self.angulos + acao[:3] * PASSO_ANGULO_MAX, LIMITES_MIN, LIMITES_MAX)
        self.garra = float(np.clip(self.garra + acao[3] * PASSO_GARRA_MAX, 0.0, 1.0))

        juntas = cinematica_direta(self.angulos)
        ponta = juntas[-1]
        distancia = float(np.linalg.norm(ponta - self.caixa))

        # 2) Física simplificada da preensão.
        pegou_agora = False
        if not self.segurando and self.garra < GARRA_FECHADA and distancia < ALCANCE_PEGAR:
            self.segurando = True
            pegou_agora = True
        elif self.segurando and self.garra > GARRA_SOLTA:
            self.segurando = False  # abriu a garra: a caixa cai

        if self.segurando:
            self.caixa = ponta.copy()  # a caixa acompanha a garra
        else:
            self.caixa[1] -= QUEDA_POR_PASSO
        self.caixa[1] = max(self.caixa[1], MEIO_LADO_CAIXA)  # não atravessa o chão

        altura = self.caixa[1] - MEIO_LADO_CAIXA

        # 3) Recompensa em etapas. Todo passo custa um pouco, então terminar
        #    rápido é melhor; e estar segurando é sempre melhor do que não.
        if self.segurando:
            recompensa = -(ALTURA_ALVO - altura)          # entre -0.4 e 0
        else:
            recompensa = -1.0 - distancia                 # sempre <= -1
        if pegou_agora and not self.ja_pegou:
            recompensa += 2.0
            self.ja_pegou = True
        recompensa -= 0.01 * float(np.sum(acao ** 2))    # evita movimentos bruscos
        if np.min(juntas[1:, 1]) < 0.0:
            recompensa -= 0.5                             # bateu no chão

        sucesso = self.segurando and altura >= ALTURA_ALVO
        if sucesso:
            recompensa += 10.0
        if self.tipo_recompensa == "esparsa":
            recompensa = 10.0 if sucesso else 0.0  # sem nenhuma pista no caminho

        terminou = bool(sucesso)
        truncou = self.passos >= MAX_PASSOS
        info = {"is_success": bool(sucesso), "distancia": distancia, "segurando": self.segurando}

        if self.render_mode == "human":
            self.render()
        return self._observacao(), float(recompensa), terminou, truncou, info

    def render(self):
        if self.render_mode is None:
            return None
        if self._tela is None:
            from render import Renderizador  # importa só se for desenhar
            self._tela = Renderizador(visivel=(self.render_mode == "human"))

        if self.render_mode == "rgb_array":
            return self._tela.capturar(self.estado_para_desenho())

        self._tela.desenhar(self.estado_para_desenho())
        # Segura a velocidade para dar tempo de ver o movimento.
        espera = 1.0 / self.metadata["render_fps"] - (time.time() - self._ultimo_quadro)
        if espera > 0:
            time.sleep(espera)
        self._ultimo_quadro = time.time()
        return None

    def close(self):
        if self._tela is not None:
            self._tela.fechar()
            self._tela = None

    # --------------------------------------------------------------
    # Auxiliares
    # --------------------------------------------------------------
    def estado_para_desenho(self):
        """Tudo que o OpenGL precisa saber para desenhar a cena."""
        return {
            "angulos": self.angulos.copy(),
            "garra": self.garra,
            "caixa": self.caixa.copy(),
            "segurando": self.segurando,
        }

    def _observacao(self):
        """O que o agente "enxerga" (números, não a imagem):
        seno e cosseno dos 3 ângulos (6), ângulos normalizados em [-1, 1] (3),
        abertura da garra (1), posição da ponta (2), posição da caixa (2),
        vetor caixa - ponta (2), se está segurando (1). Total: 17.
        """
        ponta = cinematica_direta(self.angulos)[-1]
        meio = (LIMITES_MAX + LIMITES_MIN) / 2
        metade = (LIMITES_MAX - LIMITES_MIN) / 2
        obs = np.concatenate([
            np.sin(self.angulos),
            np.cos(self.angulos),
            (self.angulos - meio) / metade,
            [self.garra],
            ponta,
            self.caixa,
            self.caixa - ponta,
            [1.0 if self.segurando else 0.0],
        ])
        return obs.astype(np.float32)
