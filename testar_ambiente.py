"""
Teste rápido, sem janela: confere se o ambiente segue o padrão do Gymnasium
e roda alguns episódios com ações aleatórias.

    python testar_ambiente.py
"""
import numpy as np
from stable_baselines3.common.env_checker import check_env

from ambiente import BracoRoboticoEnv


def main():
    env = BracoRoboticoEnv()
    check_env(env)
    print("Ambiente OK no padrão Gymnasium.")

    for ep in range(3):
        obs, _ = env.reset(seed=ep)
        retorno, fim = 0.0, False
        while not fim:
            obs, recompensa, terminou, truncou, info = env.step(env.action_space.sample())
            retorno += recompensa
            fim = terminou or truncou
        print(f"Episódio aleatório {ep + 1}: retorno {retorno:.1f}, "
              f"sucesso {info['is_success']}, observação com {obs.shape[0]} valores, "
              f"sem NaN: {not np.isnan(obs).any()}")


if __name__ == "__main__":
    main()
