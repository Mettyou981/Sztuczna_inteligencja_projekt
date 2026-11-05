import os
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from haxball_env import HaxballEnv

# ==========================================
# KONFIGURACJA TRENINGU - ZMIENIAJ TUTAJ
# ==========================================
LOAD_MODEL = "models/haxball_agent_v2.3"
SAVE_NAME = "models/haxball_agent_v2.4"
LOG_DIR = "./logs/"

# Ustaw na True żeby używać curriculum, False żeby trenować jednoetapowo.
USE_CURRICULUM = False

# Etapy curriculum
CURRICULUM_STAGES = [
    ("static", 5000_000),
    ("bot",    7000_000),
    ("random_pool",  8000_000), # Możesz teraz używać random_pool również w Curriculum
]

# Ustawienia dla trybu jednoetapowego:
OPPONENT_TYPE = "random_pool"
OPPONENT_MODEL_PATH = "models/haxball_agent_v2.3" # Sparingpartner dla trybu agent/mixed/random_pool
STEPS = 15_000_000

N_ENVS = 12
# ==========================================

def make_env(opponent_type, opponent_model=None):
    """Factory function dla SubprocVecEnv."""
    def _init():
        env = HaxballEnv(render_mode=None, opponent_type=opponent_type)
        if opponent_model is not None:
            env.set_opponent_model(opponent_model)
        return env
    return _init


def build_vec_env(opponent_type, opponent_model=None, n_envs=N_ENVS):
    env = SubprocVecEnv([make_env(opponent_type, opponent_model) for _ in range(n_envs)])
    env = VecMonitor(env, LOG_DIR)
    return env


def create_model(env):
    """Tworzy nowy model PPO z dostrojonymi hiperparametrami."""
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=5e-5,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.025,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])
        ),
        tensorboard_log="./tensorboard_logs/",
    )


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    if USE_CURRICULUM:
        print("=" * 50)
        print("TRYB CURRICULUM LEARNING")
        print("=" * 50)

        model = None
        opponent_model = None

        # DODANO: Sprawdzanie czy random_pool jest w etapach
        if any(stage[0] in ["agent", "mixed", "random_pool"] for stage in CURRICULUM_STAGES):
            try:
                print(f"Wczytywanie modelu przeciwnika: {OPPONENT_MODEL_PATH}...")
                opponent_model = PPO.load(OPPONENT_MODEL_PATH, env=None)
            except Exception as e:
                print(f"Nie znaleziono modelu przeciwnika ({e}). Etapy 'agent'/'mixed'/'random_pool' zamienione na 'bot'.")
                opponent_model = None

        for stage_num, (opp_type, stage_steps) in enumerate(CURRICULUM_STAGES, 1):
            active_opp = opp_type
            # DODANO: Zabezpieczenie dla random_pool
            if opp_type in ["agent", "mixed", "random_pool"] and opponent_model is None:
                active_opp = "bot"

            print(f"\n--- Etap {stage_num}/{len(CURRICULUM_STAGES)}: {active_opp} ({stage_steps:,} kroków) ---")

            # DODANO: Przekazywanie modelu przeciwnika dla random_pool
            needs_opp_model = active_opp in ["agent", "mixed", "random_pool"]
            env = build_vec_env(active_opp, opponent_model if needs_opp_model else None)

            if model is None:
                if LOAD_MODEL:
                    # Jeśli ładujemy bazę na start curriculum, też wymuszamy parametry
                    custom_hyperparams = {"learning_rate": 1e-4, "ent_coef": 0.025}
                    model = PPO.load(LOAD_MODEL, env=env, custom_objects=custom_hyperparams)
                else:
                    model = create_model(env)
            else:
                model.set_env(env)

            model.learn(total_timesteps=stage_steps, reset_num_timesteps=(stage_num == 1))

            stage_save = f"{SAVE_NAME}_stage{stage_num}"
            model.save(stage_save)
            print(f"Etap {stage_num} zapisany: {stage_save}.zip")

            env.close()

        model.save(SAVE_NAME)
        print(f"\nFinalny model zapisany: {SAVE_NAME}.zip")

    else:
        # ---- TRENING JEDNOETAPOWY ----
        print("TRYB JEDNOETAPOWY")
        opponent_model = None

        # DODANO: Ładowanie przeciwnika również dla random_pool
        if OPPONENT_TYPE in ["agent", "mixed", "random_pool"]:
            try:
                print(f"Wczytywanie modelu przeciwnika: {OPPONENT_MODEL_PATH}...")
                opponent_model = PPO.load(OPPONENT_MODEL_PATH, env=None)
            except Exception as e:
                print(f"Błąd: {e}. Używam bota.")

        env = build_vec_env(OPPONENT_TYPE, opponent_model)

        if LOAD_MODEL:
            print(f"Wczytywanie istniejącego modelu głównego: {LOAD_MODEL}...")
            # DODANO: Wstrzykiwanie hiperparametrów do istniejącego modelu
            custom_hyperparams = {
                "learning_rate": 1e-4,
                "ent_coef": 0.025
            }
            model = PPO.load(LOAD_MODEL, env=env, custom_objects=custom_hyperparams)
        else:
            print("Tworzenie nowego modelu od zera...")
            model = create_model(env)

        model.learn(total_timesteps=STEPS)
        model.save(SAVE_NAME)
        print(f"Model zapisany: {SAVE_NAME}.zip")
        env.close()


if __name__ == "__main__":
    main()