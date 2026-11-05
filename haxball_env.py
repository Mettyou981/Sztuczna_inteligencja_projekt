import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
import math
import random


class HaxballEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, opponent_type="bot", opponent_model=None):
        super().__init__()
        self.render_mode = render_mode
        self.width, self.height = 800, 400

        self.base_opponent_type = opponent_type
        self.active_opponent = opponent_type if opponent_type != "mixed" else "bot"
        self.opponent_model = opponent_model

        # Akcje 0-8: sam ruch, 9-17: ruch + kopnięcie
        # Kierunki: 0=stój, 1=góra, 2=dół, 3=lewo, 4=prawo, 5=góra-prawo, 6=dół-prawo, 7=góra-lewo, 8=dół-lewo
        self.action_space = spaces.Discrete(18)

        # 20 Obserwacji
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Haxball DRL - Ulepszona wersja")
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        self._setup_physics()
        self.is_kicking = False
        self.is_kicking_opp = False

        # ==========================================
        # SYSTEM NAGRÓD - KONFIGURACJA
        # ==========================================
        self.REW_TIME_PENALTY = 0.5  # Presja czasu MA BYC DODATNIE
        self.REW_DIST_BALL_MULT = 0.01  # Mnożnik za zbliżanie się do piłki
        self.REW_BALL_GOAL_MULT = 0.05  # Mnożnik radaru bramkowego (atak)
        self.REW_DEFENSE_MULT = 0.02  # Mnożnik radaru bramkowego (obrona)

        self.REW_GOAL_SCORED = 100.0  # Nagroda za strzelenie gola
        self.REW_GOAL_CONCEDED = -100.0  # Kara za stratę gola
        # ==========================================

    def set_opponent_model(self, model):
        self.opponent_model = model

    def _setup_physics(self):
        self.space = pymunk.Space()
        self.space.damping = 0.9
        PLAYER_CAT = 0b0001
        BALL_CAT = 0b0010
        WALL_CAT = 0b0100
        GOAL_BLOCKER_CAT = 0b1000

        ball_mask = pymunk.ShapeFilter.ALL_MASKS() ^ GOAL_BLOCKER_CAT
        goal_blocker_filter = pymunk.ShapeFilter(categories=GOAL_BLOCKER_CAT, mask=PLAYER_CAT)
        player_filter = pymunk.ShapeFilter(categories=PLAYER_CAT)
        ball_filter = pymunk.ShapeFilter(categories=BALL_CAT, mask=ball_mask)
        wall_filter = pymunk.ShapeFilter(categories=WALL_CAT)

        static_lines = [
            pymunk.Segment(self.space.static_body, (0, 0), (self.width, 0), 5),
            pymunk.Segment(self.space.static_body, (0, self.height), (self.width, self.height), 5),
            pymunk.Segment(self.space.static_body, (0, 0), (0, 100), 5),
            pymunk.Segment(self.space.static_body, (0, 300), (0, self.height), 5),
            pymunk.Segment(self.space.static_body, (self.width, 0), (self.width, 100), 5),
            pymunk.Segment(self.space.static_body, (self.width, 300), (self.width, self.height), 5)
        ]

        for line in static_lines:
            line.elasticity = 1.0
            line.filter = wall_filter
            self.space.add(line)

        goal_blockers = [
            pymunk.Segment(self.space.static_body, (0, 100), (0, 300), 5),
            pymunk.Segment(self.space.static_body, (self.width, 100), (self.width, 300), 5)
        ]

        for blocker in goal_blockers:
            blocker.elasticity = 0.5
            blocker.filter = goal_blocker_filter
            self.space.add(blocker)

        self.agent_body = pymunk.Body(1.5, float('inf'))
        self.agent_shape = pymunk.Circle(self.agent_body, 15)
        self.agent_shape.color = pygame.Color("red")
        self.agent_shape.elasticity = 0.5
        self.agent_shape.filter = player_filter
        self.space.add(self.agent_body, self.agent_shape)

        self.opp_body = pymunk.Body(1.5, float('inf'))
        self.opp_shape = pymunk.Circle(self.opp_body, 15)
        self.opp_shape.color = pygame.Color("blue")
        self.opp_shape.elasticity = 0.5
        self.opp_shape.filter = player_filter
        self.space.add(self.opp_body, self.opp_shape)

        moment = pymunk.moment_for_circle(1.0, 0, 10)
        self.ball_body = pymunk.Body(1.0, moment)
        self.ball_shape = pymunk.Circle(self.ball_body, 10)
        self.ball_shape.color = pygame.Color("white")
        self.ball_shape.elasticity = 0.8
        self.ball_shape.filter = ball_filter
        self.space.add(self.ball_body, self.ball_shape)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # LOSOWANIE PRZECIWNIKA CO EPIZOD RANDOM POOL
        if self.base_opponent_type == "random_pool":
            self.active_opponent = random.choices(
                ["static", "bot", "hard_bot", "agent"],
                weights=[0.05, 0.05, 0.05, 0.85]
            )[0]
        elif self.base_opponent_type == "mixed":
            self.active_opponent = random.choice(["bot", "agent"])
        else:
            self.active_opponent = self.base_opponent_type

        self.bot_kick_cooldown = 0

        agent_x = random.uniform(100, 350)
        agent_y = random.uniform(60, 340)
        opp_x = random.uniform(450, 700)
        opp_y = random.uniform(60, 340)

        ball_x = random.uniform(320, 480)
        ball_y = random.uniform(100, 300)

        self.agent_body.position = (agent_x, agent_y)
        self.agent_body.velocity = (0, 0)
        self.opp_body.position = (opp_x, opp_y)
        self.opp_body.velocity = (0, 0)
        self.ball_body.position = (ball_x, ball_y)
        self.ball_body.angular_velocity = 0

        # Losowy strzał startowy w różnych kierunkach
        wyrzut = random.uniform(80, 180)
        angle = random.uniform(0, 2 * math.pi)
        self.ball_body.velocity = (math.cos(angle) * wyrzut, math.sin(angle) * wyrzut)

        self.prev_distance_to_ball = self._get_distance(self.agent_body.position, self.ball_body.position)
        self.prev_ball_to_goal = math.hypot(800 - self.ball_body.position.x, 200 - self.ball_body.position.y)
        self.prev_opp_to_our_goal = math.hypot(self.ball_body.position.x, 200 - self.ball_body.position.y)

        self.steps = 0
        self.is_kicking = False
        self.is_kicking_opp = False

        # Statystyki epizodu
        self.episode_goals_scored = 0
        self.episode_goals_conceded = 0

        return self._get_obs(), {}

    def _apply_movement(self, body, move_action, force_multiplier):
        """Pomocnicza funkcja: aplikuje siłę do ciała na podstawie akcji ruchu (0-8)."""
        force_x, force_y = 0, 0
        DIAG = force_multiplier * 0.707  # 1/sqrt(2) dla normalizacji przekątnych

        if move_action == 1:   force_y = -force_multiplier
        elif move_action == 2: force_y = force_multiplier
        elif move_action == 3: force_x = -force_multiplier
        elif move_action == 4: force_x = force_multiplier
        elif move_action == 5: force_x, force_y = DIAG, -DIAG   # góra-prawo
        elif move_action == 6: force_x, force_y = DIAG, DIAG    # dół-prawo
        elif move_action == 7: force_x, force_y = -DIAG, -DIAG  # góra-lewo
        elif move_action == 8: force_x, force_y = -DIAG, DIAG   # dół-lewo

        body.apply_force_at_local_point((force_x, force_y), (0, 0))

    def step(self, action):
        self.steps += 1

        # Zwiększona siła, aby zrekompensować nowe, mocne hamowanie
        force_multiplier = 3500

        move_action = action % 9
        kick_action = action >= 9

        self.is_kicking = kick_action
        self.is_kicking_opp = False

        # Flagi gwarantujące, że w ciągu makro-kroku (6 klatek) agent kopnie piłkę tylko raz
        kicked_this_step = False
        opp_kicked_this_step = False

        reward = 0.0
        terminated = False
        truncated = self.steps >= 1000

        #PREDYKCJA PRZECIWNIKA (10 razy na sekundę)
        if self.active_opponent == "agent" and self.opponent_model is not None:
            opp_obs = self._get_opp_obs()
            # Używamy deterministic=True, aby przeciwnik grał "najlepiej jak potrafi" bez losowości
            opp_action, _ = self.opponent_model.predict(opp_obs, deterministic=True)

            raw_opp_move = opp_action % 9
            opp_kick = opp_action >= 9

            if self.active_opponent == "static":
                opp_move = 0
                opp_kick = False

            # MAPOWANIE LUSTRZANE AKCJI DLA NIEBIESKIEGO
            mirror_map = {
                0: 0,
                1: 1,
                2: 2,
                3: 4,
                4: 3,
                5: 7,
                6: 8,
                7: 5,
                8: 6
            }
            opp_move = mirror_map[raw_opp_move]
        else:
            opp_move = 0
            opp_kick = False

        # FRAME SKIP LOGIC (6 klatek fizyki na 1 decyzję sieci)
        for _ in range(6):
            # 1. Ruch głównego Agenta
            self._apply_movement(self.agent_body, move_action, force_multiplier)

            if kick_action and not kicked_this_step:
                dx_ball = self.ball_body.position.x - self.agent_body.position.x
                dy_ball = self.ball_body.position.y - self.agent_body.position.y
                dist_to_ball = math.hypot(dx_ball, dy_ball)
                if 0 < dist_to_ball < 35:
                    kick_strength = 220
                    nx, ny = dx_ball / dist_to_ball, dy_ball / dist_to_ball
                    self.ball_body.apply_impulse_at_local_point((nx * kick_strength, ny * kick_strength), (0, 0))
                    kicked_this_step = True

            # 2. Ruch Przeciwnika
            if self.active_opponent == "bot":
                dx = self.ball_body.position.x - self.opp_body.position.x
                dy = self.ball_body.position.y - self.opp_body.position.y
                dist = math.hypot(dx, dy)

                if self.bot_kick_cooldown > 0:
                    self.bot_kick_cooldown -= 1

                if dist > 0:
                    opp_force_x = (dx / dist) * force_multiplier * 0.8
                    opp_force_y = (dy / dist) * force_multiplier * 0.8
                    self.opp_body.apply_force_at_local_point((opp_force_x, opp_force_y), (0, 0))

                    if dist < 35 and self.bot_kick_cooldown == 0:
                        self.is_kicking_opp = True
                        if not opp_kicked_this_step:
                            vel_x = self.opp_body.velocity.x
                            vel_y = self.opp_body.velocity.y
                            speed = math.hypot(vel_x, vel_y)
                            if speed > 1.0:
                                nx_kick = vel_x / speed
                                ny_kick = vel_y / speed
                            else:
                                nx_kick = dx / dist
                                ny_kick = dy / dist
                            self.ball_body.apply_impulse_at_local_point((nx_kick * 150, ny_kick * 150), (0, 0))
                            opp_kicked_this_step = True
                            self.bot_kick_cooldown = 15

            # 2b. Ruch Przeciwnika (HARD BOT)
            elif self.active_opponent == "hard_bot":
                ball_x, ball_y = self.ball_body.position
                opp_x, opp_y = self.opp_body.position
                ball_vx, ball_vy = self.ball_body.velocity

                # 1. Przewidywanie przyszłości (Gdzie będzie piłka za ~12 klatek?)
                future_ball_x = ball_x + ball_vx * 0.2
                future_ball_y = ball_y + ball_vy * 0.2

                # Trzymamy przewidywania w granicach boiska
                future_ball_x = max(20, min(780, future_ball_x))
                future_ball_y = max(20, min(380, future_ball_y))

                # Cele: Atakujemy lewą bramkę (0, 200), Bronimy prawej (800, 200)
                goal_x, goal_y = 0, 200

                # 2. Ocena sytuacji taktycznej
                # Jeśli bot jest z lewej strony piłki (opp_x < ball_x - 20),
                # oznacza to, że został wyminiony i czerwony gracz ma pustą drogę do bramki.
                if opp_x < ball_x - 20:
                    # PANIKA: Powrót do obrony! Bot biegnie na pozycję między piłkę a swoją bramkę.
                    move_target_x = future_ball_x + 60
                    # Środek ciężkości między piłką a własną bramką w osi Y
                    move_target_y = ball_y + (200 - ball_y) * 0.5
                else:
                    # OFENSYWA: Bot jest w dobrej pozycji (po prawej stronie piłki).
                    # Wektor od przewidywanej piłki do bramki
                    to_goal_x = goal_x - future_ball_x
                    to_goal_y = goal_y - future_ball_y
                    dist_to_goal = math.hypot(to_goal_x, to_goal_y)

                    if dist_to_goal > 0:
                        dir_goal_x = to_goal_x / dist_to_goal
                        dir_goal_y = to_goal_y / dist_to_goal
                    else:
                        dir_goal_x, dir_goal_y = -1, 0

                    # Sweet Spot: 40 pikseli ZA przewidywaną piłką
                    sweet_spot_x = future_ball_x - (dir_goal_x * 40)
                    sweet_spot_y = future_ball_y - (dir_goal_y * 40)

                    dist_to_sweet_spot = math.hypot(sweet_spot_x - opp_x, sweet_spot_y - opp_y)

                    if dist_to_sweet_spot < 45:
                        # Jesteśmy w strefie uderzenia - szarża na bramkę!
                        move_target_x = goal_x
                        move_target_y = goal_y
                    else:
                        # Biegniemy do Sweet Spotu na przechwyt
                        move_target_x = sweet_spot_x
                        move_target_y = sweet_spot_y

                # 3. Aplikacja siły ruchu
                move_dx = move_target_x - opp_x
                move_dy = move_target_y - opp_y
                move_dist = math.hypot(move_dx, move_dy)

                if self.bot_kick_cooldown > 0:
                    self.bot_kick_cooldown -= 1

                if move_dist > 0:
                    # Bot używa maksymalnej siły
                    opp_force_x = (move_dx / move_dist) * force_multiplier
                    opp_force_y = (move_dy / move_dist) * force_multiplier
                    self.opp_body.apply_force_at_local_point((opp_force_x, opp_force_y), (0, 0))

                # 4. KOPNIĘCIE (W oparciu o prawdziwą pozycję, a nie przewidywaną)
                dist_to_ball = math.hypot(ball_x - opp_x, ball_y - opp_y)
                if dist_to_ball < 35 and self.bot_kick_cooldown == 0:
                    self.is_kicking_opp = True
                    if not opp_kicked_this_step:
                        # Strzał do bramki
                        kick_dx = goal_x - ball_x
                        kick_dy = goal_y - ball_y
                        kick_dist = math.hypot(kick_dx, kick_dy)

                        if kick_dist > 0:
                            self.ball_body.apply_impulse_at_local_point(
                                ((kick_dx / kick_dist) * 250, (kick_dy / kick_dist) * 250), (0, 0))

                        opp_kicked_this_step = True
                        self.bot_kick_cooldown = 15

            elif self.active_opponent == "agent" and self.opponent_model is not None:
                self._apply_movement(self.opp_body, opp_move, force_multiplier)
                if opp_kick:
                    self.is_kicking_opp = True
                    if not opp_kicked_this_step:
                        dx_ball = self.ball_body.position.x - self.opp_body.position.x
                        dy_ball = self.ball_body.position.y - self.opp_body.position.y
                        dist_to_ball = math.hypot(dx_ball, dy_ball)
                        if 0 < dist_to_ball < 35:
                            kick_strength = 220
                            nx, ny = dx_ball / dist_to_ball, dy_ball / dist_to_ball
                            self.ball_body.apply_impulse_at_local_point((nx * kick_strength, ny * kick_strength),
                                                                        (0, 0))
                            opp_kicked_this_step = True


            #Krok Fizyki (Pymunk)
            self.space.step(1 / 60.0)

            #CUSTOM FRICTION (Twarde wyhamowywanie postaci)
            # W każdej klatce ucinamy 12% prędkości graczy, co daje efekt natychmiastowego zatrzymania.
            self.agent_body.velocity = self.agent_body.velocity * 0.88
            self.opp_body.velocity = self.opp_body.velocity * 0.88

            # Ograniczenie prędkości piłki (aby nie wyleciała poza mapę)
            MAX_BALL_SPEED = 800.0
            if self.ball_body.velocity.length > MAX_BALL_SPEED:
                self.ball_body.velocity = self.ball_body.velocity.normalized() * MAX_BALL_SPEED

            # Płynne renderowanie człowieka na poziomie klatek, nie makro-kroków
            if self.render_mode == "human":
                self.render()

            # 5. Sprawdzenie Goli (Przerwanie pętli jeśli gol padnie w trakcie klatek)
            if self.ball_body.position.x > self.width:
                reward += self.REW_GOAL_SCORED
                terminated = True
                print(f"GOL DLA CZERWONEGO! (krok {self.steps}, vs {self.active_opponent})")
                break #bardzo wazna zmiana o ktorej zapomnialem

            elif self.ball_body.position.x < 0:
                reward += self.REW_GOAL_CONCEDED  # Zauważ plus, bo kara jest już zdefiniowana jako liczba ujemna (-100.0)
                terminated = True
                print(f"Niebieski strzelił gola :( (krok {self.steps}, vs {self.active_opponent})")
                break

        #KARA ZA CZAS
        reward -= self.REW_TIME_PENALTY

        """
        if not terminated:
            # 2. Zbierz postęp zbliżania się do piłki
            current_distance_to_ball = self._get_distance(self.agent_body.position, self.ball_body.position)
            dist_reward = self.REW_DIST_BALL_MULT * (self.prev_distance_to_ball - current_distance_to_ball)
            reward += dist_reward
            self.prev_distance_to_ball = current_distance_to_ball

            # 3. Radar bramkowy (postęp w kierunku bramki przeciwnika)
            current_ball_to_goal = math.hypot(800 - self.ball_body.position.x, 200 - self.ball_body.position.y)
            ball_goal_reward = self.REW_BALL_GOAL_MULT * (self.prev_ball_to_goal - current_ball_to_goal)
            reward += ball_goal_reward
            self.prev_ball_to_goal = current_ball_to_goal

            # 4. Kara za postęp piłki do naszej bramki
            current_opp_to_our_goal = math.hypot(self.ball_body.position.x, 200 - self.ball_body.position.y)
            defense_penalty = self.REW_DEFENSE_MULT * (self.prev_opp_to_our_goal - current_opp_to_our_goal)
            reward -= defense_penalty
            self.prev_opp_to_our_goal = current_opp_to_our_goal
        """

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        obs = np.zeros(20, dtype=np.float32)
        MAX_SPEED = 800.0

        # Pozycje i prędkości agenta
        obs[0] = self.agent_body.position.x / self.width
        obs[1] = self.agent_body.position.y / self.height
        obs[2] = self.agent_body.velocity.x / MAX_SPEED
        obs[3] = self.agent_body.velocity.y / MAX_SPEED

        # Pozycje i prędkości przeciwnika
        obs[4] = self.opp_body.position.x / self.width
        obs[5] = self.opp_body.position.y / self.height
        obs[6] = self.opp_body.velocity.x / MAX_SPEED
        obs[7] = self.opp_body.velocity.y / MAX_SPEED

        # Pozycja i prędkość piłki
        obs[8] = self.ball_body.position.x / self.width
        obs[9] = self.ball_body.position.y / self.height
        obs[10] = self.ball_body.velocity.x / MAX_SPEED
        obs[11] = self.ball_body.velocity.y / MAX_SPEED

        # Wektor agent -> piłka
        obs[12] = (self.ball_body.position.x - self.agent_body.position.x) / self.width
        obs[13] = (self.ball_body.position.y - self.agent_body.position.y) / self.height

        # Wektor piłka -> bramka przeciwnika
        obs[14] = (800 - self.ball_body.position.x) / self.width
        obs[15] = (200 - self.ball_body.position.y) / self.height

        # NOWE: Wektor przeciwnik -> piłka (pomaga agentowi przewidywać ruchy)
        obs[16] = (self.ball_body.position.x - self.opp_body.position.x) / self.width
        obs[17] = (self.ball_body.position.y - self.opp_body.position.y) / self.height

        # NOWE: Czy agent jest blisko piłki? (0 lub 1 — binary cue)
        dist_to_ball = self._get_distance(self.agent_body.position, self.ball_body.position)
        obs[18] = 1.0 if dist_to_ball < 35 else 0.0

        # NOWE: Postęp czasowy epizodu (0 -> 1) — agent wie ile czasu zostało
        obs[19] = self.steps / 1000.0

        return obs

    def _get_opp_obs(self):
        """Obserwacja z perspektywy przeciwnika (zwierciadło lustrzane)."""
        obs = np.zeros(20, dtype=np.float32)
        MAX_SPEED = 800.0

        obs[0] = (self.width - self.opp_body.position.x) / self.width
        obs[1] = self.opp_body.position.y / self.height
        obs[2] = -self.opp_body.velocity.x / MAX_SPEED
        obs[3] = self.opp_body.velocity.y / MAX_SPEED

        obs[4] = (self.width - self.agent_body.position.x) / self.width
        obs[5] = self.agent_body.position.y / self.height
        obs[6] = -self.agent_body.velocity.x / MAX_SPEED
        obs[7] = self.agent_body.velocity.y / MAX_SPEED

        obs[8] = (self.width - self.ball_body.position.x) / self.width
        obs[9] = self.ball_body.position.y / self.height
        obs[10] = -self.ball_body.velocity.x / MAX_SPEED
        obs[11] = self.ball_body.velocity.y / MAX_SPEED

        obs[12] = obs[8] - obs[0]
        obs[13] = obs[9] - obs[1]

        obs[14] = obs[8]  # odległość lustrzana do bramki
        obs[15] = (200 - self.ball_body.position.y) / self.height

        obs[16] = obs[8] - obs[4]
        obs[17] = obs[9] - obs[5]

        dist_to_ball = self._get_distance(self.opp_body.position, self.ball_body.position)
        obs[18] = 1.0 if dist_to_ball < 35 else 0.0

        obs[19] = self.steps / 1000.0

        return obs

    def _get_distance(self, pos1, pos2):
        return math.sqrt((pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2)

    def render(self):
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            self.screen.fill((113, 140, 90))
            s_left = pygame.Surface((30, 200), pygame.SRCALPHA)
            s_left.fill((255, 0, 0, 50))
            self.screen.blit(s_left, (0, 100))
            s_right = pygame.Surface((30, 200), pygame.SRCALPHA)
            s_right.fill((0, 0, 255, 50))
            self.screen.blit(s_right, (self.width - 30, 100))
            pygame.draw.line(self.screen, (255, 255, 255), (2, 100), (2, 300), 4)
            pygame.draw.line(self.screen, (255, 255, 255), (self.width - 3, 100), (self.width - 3, 300), 4)
            self.space.debug_draw(self.draw_options)
            posts = [(0, 100), (0, 300), (self.width, 100), (self.width, 300)]
            for post in posts:
                pygame.draw.circle(self.screen, (255, 255, 255), post, 8)
                pygame.draw.circle(self.screen, (0, 0, 0), post, 8, 2)
            if self.is_kicking:
                pygame.draw.circle(self.screen, (255, 255, 255),
                                   (int(self.agent_body.position.x), int(self.agent_body.position.y)), 18, 3)
            if self.is_kicking_opp:
                pygame.draw.circle(self.screen, (255, 255, 255),
                                   (int(self.opp_body.position.x), int(self.opp_body.position.y)), 18, 3)
            pygame.display.flip()
            self.clock.tick(60)