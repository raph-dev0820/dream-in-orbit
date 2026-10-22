from torch import nn


class Modele(nn.Module):
    def __init__(self, n_x, n_y, cache=64):
        super().__init__()
        # le GRU : il lit les entrées de l'heure + la consommation de l'heure d'avant
        self.gru = nn.GRU(n_x + n_y, cache, batch_first=True)
        # une couche qui transforme les "notes" du GRU en 33 consommations
        self.sortie = nn.Linear(cache, n_y)

    def forward(self, x, h=None):
        z, h = self.gru(x, h)          # h = les notes (la mémoire)
        return self.sortie(z), h