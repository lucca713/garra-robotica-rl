"""
Grava o braço em um GIF animado (para slides, apresentação e artigo).

Exemplos:
    # modelo que deu certo
    python gravar.py --algo sac --modelo modelos/sac_seed0/best_model.zip --saida videos/sac_certo.gif
    # treino que deu errado (recompensa esparsa)
    python gravar.py --algo sac --modelo modelos/sac_esparsa_seed0/final.zip --saida videos/sac_errado.gif
    # braço sem treino nenhum (ações aleatórias)
    python gravar.py --aleatorio --saida videos/aleatorio.gif

Use a mesma --seed nos vídeos para as caixas aparecerem nos mesmos lugares
e a comparação ficar justa.
"""
import argparse
import os

from PIL import Image, ImageDraw
from stable_baselines3 import PPO, SAC

from ambiente import BracoRoboticoEnv


def main():
    parser = argparse.ArgumentParser(description="Grava episódios do braço em GIF.")
    parser.add_argument("--algo", choices=["ppo", "sac"])
    parser.add_argument("--modelo", help="caminho do .zip treinado")
    parser.add_argument("--aleatorio", action="store_true", help="sem modelo: ações aleatórias")
    parser.add_argument("--saida", default="videos/braco.gif")
    parser.add_argument("--episodios", type=int, default=3)
    parser.add_argument("--max-passos", type=int, default=80, help="corta episódios longos no vídeo")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--largura", type=int, default=640, help="largura do GIF em pixels")
    args = parser.parse_args()

    if not args.aleatorio and not (args.algo and args.modelo):
        parser.error("informe --algo e --modelo, ou use --aleatorio")

    modelo = None
    if not args.aleatorio:
        modelo = (PPO if args.algo == "ppo" else SAC).load(args.modelo)
        rotulo = f"{args.algo.upper()} - {os.path.basename(os.path.dirname(args.modelo))}"
    else:
        rotulo = "Sem treino (acoes aleatorias)"

    env = BracoRoboticoEnv(render_mode="rgb_array")
    quadros = []

    def guardar(texto):
        imagem = Image.fromarray(env.render())
        altura = int(imagem.height * args.largura / imagem.width)
        imagem = imagem.resize((args.largura, altura))
        ImageDraw.Draw(imagem).text((10, 10), texto, fill=(20, 20, 20))
        quadros.append(imagem)

    for ep in range(args.episodios):
        obs, _ = env.reset(seed=args.seed + ep)
        resultado = "..."
        for passo in range(args.max_passos):
            guardar(f"{rotulo} | tentativa {ep + 1} | passo {passo}")
            if modelo is None:
                acao = env.action_space.sample()
            else:
                acao, _ = modelo.predict(obs, deterministic=True)
            obs, _, terminou, truncou, info = env.step(acao)
            if terminou or truncou:
                break
        resultado = "SUCESSO" if info["is_success"] else "FALHOU"
        print(f"Tentativa {ep + 1}: {resultado}")
        for _ in range(args.fps):  # ~1 segundo parado mostrando o resultado
            guardar(f"{rotulo} | tentativa {ep + 1} | {resultado}")

    env.close()
    pasta = os.path.dirname(args.saida)
    if pasta:
        os.makedirs(pasta, exist_ok=True)
    quadros[0].save(args.saida, save_all=True, append_images=quadros[1:],
                    duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"Vídeo salvo em {args.saida} ({len(quadros)} quadros)")


if __name__ == "__main__":
    main()
