"""
Carrega um modelo treinado e mostra o braço trabalhando na janela OpenGL.

Exemplos:
    python avaliar.py --algo sac --modelo modelos/sac_seed0/best_model.zip
    python avaliar.py --algo ppo --modelo modelos/ppo_seed0/best_model.zip --episodios 100 --sem-janela

No final imprime a taxa de sucesso, que vai para a seção de Resultados do artigo.
"""
import argparse

import numpy as np
from stable_baselines3 import PPO, SAC

from ambiente import BracoRoboticoEnv


def main():
    parser = argparse.ArgumentParser(description="Avalia um modelo treinado.")
    parser.add_argument("--algo", choices=["ppo", "sac"], required=True)
    parser.add_argument("--modelo", required=True, help="caminho do arquivo .zip")
    parser.add_argument("--episodios", type=int, default=10)
    parser.add_argument("--sem-janela", action="store_true", help="não abre o OpenGL (mais rápido)")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=10, help="velocidade da animação (quadros por segundo)")
    args = parser.parse_args()

    classe = PPO if args.algo == "ppo" else SAC
    modelo = classe.load(args.modelo)
    env = BracoRoboticoEnv(render_mode=None if args.sem_janela else "human")
    env.metadata = dict(env.metadata, render_fps=args.fps)

    sucessos, retornos, duracoes = [], [], []
    for ep in range(args.episodios):
        obs, _ = env.reset(seed=args.seed + ep)
        retorno, fim = 0.0, False
        while not fim:
            # deterministic=True: usa a melhor ação, sem o ruído de exploração do treino
            acao, _ = modelo.predict(obs, deterministic=True)
            obs, recompensa, terminou, truncou, info = env.step(acao)
            retorno += recompensa
            fim = terminou or truncou
            if env._tela is not None and not env._tela.aberta:
                env.close()
                return
        # Segura a cena final por ~1 segundo para dar tempo de ver o resultado.
        if env._tela is not None:
            for _ in range(args.fps):
                env.render()
        sucessos.append(info["is_success"])
        retornos.append(retorno)
        duracoes.append(env.passos)
        print(f"Episódio {ep + 1}: {'SUCESSO' if info['is_success'] else 'falhou'} "
              f"| retorno {retorno:.1f} | {env.passos} passos")

    env.close()
    print(f"\nTaxa de sucesso: {100 * np.mean(sucessos):.1f}% em {args.episodios} episódios")
    print(f"Retorno médio: {np.mean(retornos):.2f} (desvio {np.std(retornos):.2f})")
    print(f"Passos médios por episódio: {np.mean(duracoes):.1f}")


if __name__ == "__main__":
    main()
