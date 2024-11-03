import matplotlib.pyplot as plt
import numpy as np

# Data exacta nu poate fi calculata doar
# pe baza semnalului deoarece nu exista
# niciun reper in timp - de inceput sau
# de sfarsit.

# Totusi, ziua / ora, pot fi determinate
# fie cu ajutorul unui model ML antrenat pe date
# ori analizand manual datele.
# Putem presupune in mod euristic ca
# cele mai aglomerate perioade sunt luni, vineri
# si probabil in weekend, la orele de varf,
# seara sau in jur de ora 16:00.
# Stim ca avem ~ 2 ani de date, deci
# putem alegem un sample de la care
# incepe exact un an, iar apoi
# sa facem reverse engineering
# pentru a ne da seama daca primul sample
# este o zi exacta si o ora aproximativa.
# Metoda este extrem de aproximativa,
# deoarece outliere precum zile de
# sarbatori, zile libere etc. vor da date gresite.

if __name__ == '__main__':
    pass