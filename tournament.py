import os
import pandas as pd
from stable_baselines3 import PPO
from haxball_env import HaxballEnv
import itertools

# ==========================================
# KONFIGURACJA TURNIEJU
# ==========================================
# Lista modeli do przetestowania (możesz podać dokładne ścieżki lub "bot")
COMPETITORS = [
    "bot",
    "models/haxball_agent_v2.2",
    "models/haxball_agent_v20",
    "models/haxball_agent_v18",
    "models/haxball_agent_v17",
    "models/haxball_agent_v2.1",
    "models/haxball_agent_v2.3",
    "models/haxball_agent_v2.4"
]

MATCHES_PER_DUEL = 500  # Liczba meczów w każdym starciu (np. A vs B zagrają 100 razy)


# ==========================================

def get_short_name(path):
    """Zmienia długą ścieżkę na ładną, krótką nazwę do tabeli"""
    if path == "bot": return "BOT"
    if path == "hard_bot": return "HBOT"  # <--- Dodano skrót
    return path.split("/")[-1].replace("haxball_agent_", "")


def main():
    print("=" * 60)
    print("MISTRZOSTWA HAXBALL DRL - TRYB LIGOWY (HEADLESS)")
    print(f"Liczba meczów na pojedynek: {MATCHES_PER_DUEL}")
    print("=" * 60)

    # Inicjalizacja statystyk dla każdego zawodnika
    stats = {c: {"Punkty": 0, "Wygrane": 0, "Remisy": 0, "Przegrane": 0, "Strzelone": 0, "Stracone": 0} for c in
             COMPETITORS}

    # Tabela krzyżowa (macierz wyników kto z kim i jaki bilans)
    cross_table = {c1: {c2: "-" for c2 in COMPETITORS} for c1 in COMPETITORS}

    # Wczytywanie modeli do pamięci z góry (oszczędza czas)
    loaded_models = {}
    for c in COMPETITORS:
        if c != "bot" and c!= "hard_bot":
            try:
                loaded_models[c] = PPO.load(c, env=None)
            except Exception as e:
                print(f"Błąd ładowania modelu {c}. Upewnij się, że plik istnieje. ({e})")
                return

    # Inicjalizacja środowiska bez grafiki (render_mode=None = błyskawiczne symulacje)
    env = HaxballEnv(render_mode=None, opponent_type="bot")

    # Wygenerowanie wszystkich unikalnych par zawodników
    pairs = list(itertools.combinations(COMPETITORS, 2))

    for comp1, comp2 in pairs:
        # Pomiń mecze pomiędzy dowolnymi skryptowanymi botami
        if comp1 in ["bot", "hard_bot"] and comp2 in ["bot", "hard_bot"]:
            continue

        print(f"\nRozpoczynam pojedynek: {get_short_name(comp1)} vs {get_short_name(comp2)}")

        comp1_wins, comp2_wins, draws = 0, 0, 0
        comp1_goals, comp2_goals = 0, 0

        matchups = []
        if comp1 in ["bot", "hard_bot"]:
            matchups = [(comp2, comp1)] * MATCHES_PER_DUEL
        elif comp2 in ["bot", "hard_bot"]:
            matchups = [(comp1, comp2)] * MATCHES_PER_DUEL
        else:
            half = MATCHES_PER_DUEL // 2
            matchups = [(comp1, comp2)] * half + [(comp2, comp1)] * (MATCHES_PER_DUEL - half)

        for i, (red_player, blue_player) in enumerate(matchups):
            # Konfiguracja Niebieskiego
            if blue_player in ["bot", "hard_bot"]:
                env.active_opponent = blue_player
                env.opponent_model = None
            else:
                env.active_opponent = "agent"
                env.set_opponent_model(loaded_models[blue_player])

            # Konfiguracja Czerwonego
            red_model = loaded_models[red_player]

            obs, _ = env.reset()
            done = False

            while not done:
                # Opcja deterministic=True sprawia, że agenty grają optymalnie i nie eksperymentują
                action, _ = red_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)

                if terminated or truncated:
                    done = True
                    if terminated:
                        # W środowisku nagrody za gola są o wiele wyższe (+100/-100)
                        # niż kara za czas, więc reward > 0 jednoznacznie określa zwycięstwo Czerwonego
                        if reward > 0:
                            if red_player == comp1:
                                comp1_wins += 1;
                                comp1_goals += 1
                            else:
                                comp2_wins += 1;
                                comp2_goals += 1
                        else:  # Zwycięstwo Niebieskiego
                            if blue_player == comp1:
                                comp1_wins += 1;
                                comp1_goals += 1
                            else:
                                comp2_wins += 1;
                                comp2_goals += 1
                    else:  # Remis (minęło 1000 kroków bez gola)
                        draws += 1

            # Wizualny pasek postępu co 20%
            progress_step = max(1, MATCHES_PER_DUEL // 5)
            if (i + 1) % progress_step == 0:
                print(f"  Rozegrano {i + 1}/{MATCHES_PER_DUEL}...")

        # Aktualizacja statystyk Comp1 (Zwycięstwo: 3 pkt, Remis: 1 pkt)
        stats[comp1]["Wygrane"] += comp1_wins
        stats[comp1]["Przegrane"] += comp2_wins
        stats[comp1]["Remisy"] += draws
        stats[comp1]["Punkty"] += (comp1_wins * 3) + draws
        stats[comp1]["Strzelone"] += comp1_goals
        stats[comp1]["Stracone"] += comp2_goals

        # Aktualizacja statystyk Comp2
        stats[comp2]["Wygrane"] += comp2_wins
        stats[comp2]["Przegrane"] += comp1_wins
        stats[comp2]["Remisy"] += draws
        stats[comp2]["Punkty"] += (comp2_wins * 3) + draws
        stats[comp2]["Strzelone"] += comp2_goals
        stats[comp2]["Stracone"] += comp1_goals

        # Aktualizacja Macierzy Wyników (Zwycięstwa - Remisy - Porażki)
        n1 = get_short_name(comp1)
        n2 = get_short_name(comp2)
        cross_table[comp1][comp2] = f"{comp1_wins}-{draws}-{comp2_wins}"
        cross_table[comp2][comp1] = f"{comp2_wins}-{draws}-{comp1_wins}"

        print(f"  Końcowy wynik: {n1} {comp1_wins} - {draws} - {comp2_wins} {n2}")

    # ==========================================
    # WYSWIETLANIE WYNIKÓW (PANDAS)
    # ==========================================
    print("\n\n" + "=" * 60)
    print("TABELA LIGOWA (Sortowana wg Punktów)")
    print("=" * 60)

    df_stats = pd.DataFrame.from_dict(stats, orient='index')
    df_stats.index = [get_short_name(idx) for idx in df_stats.index]
    df_stats["Bilans"] = df_stats["Strzelone"] - df_stats["Stracone"]

    # Sortowanie: 1. Punkty, 2. Bilans bramek, 3. Ilość strzelonych
    df_stats = df_stats.sort_values(by=["Punkty", "Bilans", "Strzelone"], ascending=[False, False, False])
    print(df_stats.to_string())

    print("\n\n" + "=" * 60)
    print("TABELA KRZYŻOWA (H2H)")
    print("Jak czytać: Wiersz określa zawodnika. Format: Wygrane - Remisy - Porażki z rywalem z kolumny.")
    print("=" * 60)

    df_cross = pd.DataFrame(cross_table)
    df_cross.index = [get_short_name(idx) for idx in df_cross.index]
    df_cross.columns = [get_short_name(col) for col in df_cross.columns]
    print(df_cross.to_string())


if __name__ == "__main__":
    main()