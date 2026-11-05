def policz_czas_treningu(kroki, fps):
    # Całkowity czas w sekundach
    calkowite_sekundy = kroki / fps

    # Obliczanie poszczególnych jednostek czasu
    lata, reszta = divmod(calkowite_sekundy, 60 * 60 * 24 * 365)
    dni, reszta = divmod(reszta, 60 * 60 * 24)
    godziny, reszta = divmod(reszta, 60 * 60)
    minuty, sekundy = divmod(reszta, 60)

    print(f"Zakładane FPS: {fps}")
    print(f"Docelowa liczba kroków: {kroki:,}".replace(',', ' '))
    print("-" * 40)
    print(f"Szacowany czas treningu to:")
    print(f"{int(lata):,} lat".replace(',', ' '))
    print(f"{int(dni)} dni")
    print(f"{int(godziny)} godzin")
    print(f"{int(minuty)} minut")
    print(f"{int(sekundy)} sekund")

# Podstawiamy Twoje dane z konsoli
docelowe_kroki = 1_000_000
twoje_fps =800

policz_czas_treningu(docelowe_kroki, twoje_fps)