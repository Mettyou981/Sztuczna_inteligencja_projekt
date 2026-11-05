from stable_baselines3 import PPO
from haxball_env import HaxballEnv
import pygame
import time

# ==========================================
# KONFIGURACJA TESTU - ZMIENIAJ TUTAJ
# ==========================================
OPPONENT_TYPE = "agent"  # "static", "bot","hard_bot", lub "agent"
OPPONENT_MODEL_PATH = "models/haxball_agent_v2.4"


# ==========================================

def get_human_action():
    """Mapuje wciśnięte klawisze na przestrzeń akcji (0-17)"""
    keys = pygame.key.get_pressed()

    move_action = 0
    # Przekątne
    if keys[pygame.K_w] and keys[pygame.K_d]:
        move_action = 5
    elif keys[pygame.K_s] and keys[pygame.K_d]:
        move_action = 6
    elif keys[pygame.K_w] and keys[pygame.K_a]:
        move_action = 7
    elif keys[pygame.K_s] and keys[pygame.K_a]:
        move_action = 8
    # Proste kierunki
    elif keys[pygame.K_w]:
        move_action = 1
    elif keys[pygame.K_s]:
        move_action = 2
    elif keys[pygame.K_a]:
        move_action = 3
    elif keys[pygame.K_d]:
        move_action = 4

    # Spacja aktywuje kopnięcie (+9 do akcji)
    kick_action = 9 if keys[pygame.K_SPACE] else 0

    return move_action + kick_action


def main():
    # Render musi być "human", żeby działało okno i klawiatura
    env = HaxballEnv(render_mode="human", opponent_type=OPPONENT_TYPE)

    if OPPONENT_TYPE == "agent":
        try:
            print(f"Wczytywanie modelu przeciwnika (Niebieski): {OPPONENT_MODEL_PATH}...")
            opponent_model = PPO.load(OPPONENT_MODEL_PATH, env=None)
            env.set_opponent_model(opponent_model)
        except Exception as e:
            print(f"Błąd ładowania przeciwnika: {e}.")
            return

    obs, info = env.reset()
    episodes = 0
    total_reward = 0.0

    print("==================================================")
    print("GRAJ JAKO AGENT (CZERWONY)")
    print("Sterowanie: W, A, S, D (Ruch) | SPACJA (Kopnięcie)")
    print("Obserwuj konsolę, aby widzieć przyznawane nagrody!")
    print("==================================================")

    while True:
        # Nasłuch zdarzeń Pygame (konieczne, by okno nie "zamarzło")
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return

        # Pobranie akcji od gracza
        action = get_human_action()

        # Wykonanie kroku w środowisku
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Wypisywanie nagrody, jeśli jest znacząca (pomija stałą karę za czas)
        if abs(reward + 0.01) > 0.001:
            print(f"Krok {env.steps} | Nagroda za krok: {reward:.4f} | Suma: {total_reward:.2f}")

        if terminated or truncated:
            episodes += 1
            print(f"\nKoniec epizodu {episodes}! Całkowita nagroda: {total_reward:.2f}\n")
            obs, info = env.reset()
            total_reward = 0.0
            time.sleep(1)  # Chwila pauzy przed nowym startem


if __name__ == "__main__":
    main()