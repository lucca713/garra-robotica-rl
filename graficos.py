"""
Gera os gráficos de comparação PPO x SAC para o artigo, a partir dos
arquivos modelos/<algo>_seed<N>/evaluations.npz criados por treinar.py.

    python graficos.py

Saída: graficos/curvas.png (retorno médio e taxa de sucesso x passos),
com a média entre as sementes e uma faixa de desvio-padrão.
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def carregar(algo):
    execucoes = []
    for arquivo in sorted(glob.glob(os.path.join("modelos", f"{algo}_seed*", "evaluations.npz"))):
        dados = np.load(arquivo)
        execucoes.append((dados["timesteps"], dados["results"].mean(axis=1),
                          dados["successes"].mean(axis=1) * 100))
    if not execucoes:
        return None
    # Corta todas as sementes no mesmo tamanho para poder tirar a média.
    n = min(len(e[0]) for e in execucoes)
    passos = execucoes[0][0][:n]
    retornos = np.array([e[1][:n] for e in execucoes])
    sucessos = np.array([e[2][:n] for e in execucoes])
    return passos, retornos, sucessos, len(execucoes)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    # Os treinos com recompensa esparsa aparecem tracejados e só no gráfico de
    # sucesso, porque o retorno deles está em outra escala.
    series = (("ppo", "tab:blue", "-"), ("sac", "tab:orange", "-"),
              ("ppo_esparsa", "tab:blue", "--"), ("sac_esparsa", "tab:orange", "--"))
    for algo, cor, estilo in series:
        dados = carregar(algo)
        if dados is None:
            print(f"Nenhum resultado de {algo.upper()} encontrado.")
            continue
        passos, retornos, sucessos, n = dados
        rotulo = f"{algo.upper().replace('_', ' ')} ({n} semente{'s' if n > 1 else ''})"
        paineis = ((ax2, sucessos),) if "esparsa" in algo else ((ax1, retornos), (ax2, sucessos))
        for ax, valores in paineis:
            media, desvio = valores.mean(axis=0), valores.std(axis=0)
            ax.plot(passos, media, color=cor, linestyle=estilo, label=rotulo)
            ax.fill_between(passos, media - desvio, media + desvio, color=cor, alpha=0.2)
        print(f"{algo.upper()}: sucesso final {sucessos[:, -1].mean():.1f}% "
              f"| retorno final {retornos[:, -1].mean():.2f}")

    ax1.set(title="Retorno médio por episódio", xlabel="Passos de treino", ylabel="Retorno")
    ax2.set(title="Taxa de sucesso", xlabel="Passos de treino", ylabel="Sucesso (%)", ylim=(-5, 105))
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    os.makedirs("graficos", exist_ok=True)
    fig.savefig(os.path.join("graficos", "curvas.png"), dpi=150)
    print("Gráfico salvo em graficos/curvas.png")


if __name__ == "__main__":
    main()
