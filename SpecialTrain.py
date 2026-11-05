import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from haxball_env import HaxballEnv

LOG_DIR = "./logs/"
N_ENVS = 12


def make_env(opponent_type, opponent_model_path=None):
    def _init():
        env = HaxballEnv(render_mode=None, opponent_type=opponent_type)
        if opponent_model_path:
            opp_model = PPO.load(opponent_model_path, env=None)
            env.set_opponent_model(opp_model)
        return env

    return _init


def build_vec_env(opponent_type, opponent_model_path=None, n_envs=N_ENVS):
    env = SubprocVecEnv([make_env(opponent_type, opponent_model_path) for _ in range(n_envs)])
    env = VecMonitor(env, LOG_DIR)
    return env


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # =========================================================================
    # HARMONOGRAM TRENINGU: (Baza, Typ Wroga, Ścieżka Wroga, Nazwa Celu, Kroki)
    # =========================================================================
    stages = [
        # Etap 1: Kalibracja na bocie po naprawie mapowania akcji
        ("models/haxball_agent_v17", "bot", None, "models/haxball_agent_v18", 1_000_000),

        # Etap 2-5: Prawdziwy Self-Play (Iteracyjna ewolucja)
        ("models/haxball_agent_v18", "agent", "models/haxball_agent_v18", "models/haxball_agent_v19", 2_000_000),
        ("models/haxball_agent_v19", "agent", "models/haxball_agent_v19", "models/haxball_agent_v20", 2_000_000),
        ("models/haxball_agent_v20", "agent", "models/haxball_agent_v20", "models/haxball_agent_v21", 2_000_000),
        ("models/haxball_agent_v21", "agent", "models/haxball_agent_v21", "models/haxball_agent_v22", 2_000_000),
    ]

    print("=" * 60)
    print("ROZPOCZĘCIE AUTOMATYCZNEGO TRENINGU ITERACYJNEGO")
    print("=" * 60)

    for i, stage in enumerate(stages):
        base_model_path, opp_type, opp_path, save_name, steps = stage

        print(f"\n--- SESJA {i + 1}/{len(stages)} ---")
        print(f"Uczeń:      {base_model_path}")
        print(f"Przeciwnik: {opp_type} ({opp_path if opp_path else 'Standardowy'})")
        print(f"Zapis do:   {save_name}")
        print(f"Kroki:      {steps:,}")

        # 1. Budowa środowiska ze specyficznym przeciwnikiem dla tej sesji
        env = build_vec_env(opp_type, opp_path)

        # 2. Modyfikacja Hiperparametrów w locie (Fine-Tuning)
        custom_hyperparams = {
            "learning_rate": 1e-4,  # Niższy krok nauki zapobiega zapominaniu
            "ent_coef": 0.015  # Umiarkowana ciekawość do znajdywania luk w obronie
        }

        print("Wczytywanie i nadpisywanie hiperparametrów...")
        model = PPO.load(
            base_model_path,
            env=env,
            custom_objects=custom_hyperparams
        )

        # 3. Uruchomienie nauki
        # reset_num_timesteps=False sprawia, że logi w konsoli są ciągłe
        model.learn(total_timesteps=steps, reset_num_timesteps=False)

        # 4. Zapisanie ewolucji agenta
        model.save(save_name)
        print(f">>> ZAKOŃCZONO SESJĘ. Model zapisany jako: {save_name}.zip\n")

        env.close()

    print("Wszystkie etapy treningu zakończone sukcesem!")


if __name__ == "__main__":
    main()