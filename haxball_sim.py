import pygame
import pymunk
import pymunk.pygame_util


def main():
    # 1. Inicjalizacja Pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 400))
    pygame.display.set_caption("Haxball DRL - Symulator Fizyki")
    clock = pygame.time.Clock()
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    # 2. Inicjalizacja Pymunk (Przestrzeń fizyczna)
    space = pymunk.Space()
    space.gravity = (0, 0)  # W Haxballu patrzymy z góry, więc grawitacja to 0,0
    space.damping = 0.9  # Tarcie (symuluje zwalnianie piłki i graczy na trawie)

    # 3. Tworzenie piłki
    ball_mass = 1
    ball_radius = 15
    ball_moment = pymunk.moment_for_circle(ball_mass, 0, ball_radius)
    ball_body = pymunk.Body(ball_mass, ball_moment)
    ball_body.position = (400, 200)  # Środek boiska

    ball_shape = pymunk.Circle(ball_body, ball_radius)
    ball_shape.elasticity = 0.8  # Sprężystość (jak mocno się odbija)
    ball_shape.color = pygame.Color("white")
    space.add(ball_body, ball_shape)

    # Nadajmy piłce początkową prędkość, żeby sprawdzić czy działa
    ball_body.velocity = (300, 150)

    # 4. Tworzenie band (Ścian boiska)
    static_lines = [
        pymunk.Segment(space.static_body, (0, 0), (800, 0), 5),  # Góra
        pymunk.Segment(space.static_body, (800, 0), (800, 400), 5),  # Prawa
        pymunk.Segment(space.static_body, (800, 400), (0, 400), 5),  # Dół
        pymunk.Segment(space.static_body, (0, 400), (0, 0), 5)  # Lewa
    ]
    for line in static_lines:
        line.elasticity = 1.0  # Ściany idealnie odbijają
        space.add(line)

    # 5. Główna pętla gry
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Odświeżanie tła (kolor murawy z prezentacji)
        screen.fill((113, 140, 90))

        # Wykonanie kroku fizyki (60 klatek na sekundę)
        space.step(1 / 60.0)

        # Rysowanie wszystkich obiektów z Pymunk
        space.debug_draw(draw_options)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()