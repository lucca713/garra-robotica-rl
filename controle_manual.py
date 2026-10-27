"""
Controle o braço pelo teclado para testar a simulação e o desenho OpenGL
antes de treinar. Rode:  python controle_manual.py

    Q / A  -> ombro          W / S -> cotovelo       E / D -> punho
    F      -> fecha a garra   O    -> abre a garra
    R      -> recomeça        Setas, Z, X -> câmera
"""
import glfw
import numpy as np

from ambiente import BracoRoboticoEnv

TECLAS_JUNTAS = [
    (glfw.KEY_Q, glfw.KEY_A),
    (glfw.KEY_W, glfw.KEY_S),
    (glfw.KEY_E, glfw.KEY_D),
]


def main():
    env = BracoRoboticoEnv(render_mode="human")
    env.reset(seed=0)
    tela = env._tela
    print(__doc__)

    while tela.aberta:
        acao = np.zeros(4, dtype=np.float32)
        for i, (mais, menos) in enumerate(TECLAS_JUNTAS):
            if tela.tecla_pressionada(mais):
                acao[i] = 1.0
            if tela.tecla_pressionada(menos):
                acao[i] = -1.0
        if tela.tecla_pressionada(glfw.KEY_F):
            acao[3] = -1.0
        if tela.tecla_pressionada(glfw.KEY_O):
            acao[3] = 1.0
        if tela.tecla_pressionada(glfw.KEY_R):
            env.reset()
            continue

        _, recompensa, terminou, truncou, info = env.step(acao)
        print(f"\rrecompensa {recompensa:6.2f} | distância {info['distancia']:.3f} "
              f"| segurando {info['segurando']!s:5}", end="")
        if terminou or truncou:
            print("\n" + ("SUCESSO! Caixa levantada." if info["is_success"] else "Tempo esgotado."))
            env.reset()

    env.close()


if __name__ == "__main__":
    main()
