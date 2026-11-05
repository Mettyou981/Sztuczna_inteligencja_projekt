from stable_baselines3 import PPO
from haxball_env import HaxballEnv
import time

# ==========================================
# KONFIGURACJA TESTU - ZMIENIAJ TUTAJ
# ==========================================
MAIN_MODEL_PATH = "models/haxball_agent_v2.4"
OPPONENT_TYPE = "agent"  # "static", "bot","hard_bot" lub "agent"
OPPONENT_MODEL_PATH = "models/haxball_agent_v17"
# ==========================================

def main():
    env = HaxballEnv(render_mode="human", opponent_type=OPPONENT_TYPE)

    if OPPONENT_TYPE == "agent":
        try:
            print(f"Wczytywanie modelu przeciwnika (Niebieski): {OPPONENT_MODEL_PATH}...")
            opponent_model = PPO.load(OPPONENT_MODEL_PATH, env=None)
            env.set_opponent_model(opponent_model)
        except Exception as e:
            print(f"Błąd ładowania przeciwnika: {e}.")
            return

    print(f"Wczytywanie modelu głównego (Czerwony): {MAIN_MODEL_PATH}...")
    model = PPO.load(MAIN_MODEL_PATH)

    obs, info = env.reset()

    # Statystyki
    goals_scored = 0
    goals_conceded = 0
    episodes = 0

    print(f"Rozpoczęcie gry testowej! Czerwony vs {OPPONENT_TYPE}")
    for _ in range(5000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            episodes += 1
            if terminated:
                if reward > 0:
                    goals_scored += 1
                else:
                    goals_conceded += 1
            print(f"Epizod {episodes}: {goals_scored} - {goals_conceded}")
            obs, info = env.reset()
            time.sleep(0.5)

if __name__ == "__main__":
    main()