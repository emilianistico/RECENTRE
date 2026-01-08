import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns

def calcola_fd(angoli):
    angoli = np.array(angoli)
    diff = np.diff(angoli, axis=0)
    diff[:, 3:6] *= 50  # rotazioni in mm
    fd = np.sum(np.abs(diff), axis=1)
    return fd

def parse_array_string(array_str):
    righe = array_str.strip().split('\n')
    return np.array([[float(x) for x in riga.strip().split()] for riga in righe])

def processa_dataset(dataset):
    """
    Per ogni paziente ritorna: picco, media, std del Framewise Displacement.
    """
    picchi_fd, medie_fd, std_fd = [], [], []
    for stringa in dataset:
        array_dati = parse_array_string(stringa)
        angoli = array_dati[:, 1:]  # Esclude eventuale ID/tempo
        fd = calcola_fd(angoli)
        picchi_fd.append(np.max(fd))
        medie_fd.append(np.mean(fd))
        std_fd.append(np.std(fd))
    return np.array(picchi_fd), np.array(medie_fd), np.array(std_fd)

def visualizza_metriche(metric1, metric2, nomi_file, nome_metrica):
    """
    Confronta due metriche (picco, media, std) tra due dataset.
    """
    rank1 = np.argsort(np.argsort(-metric1))
    rank2 = np.argsort(np.argsort(-metric2))
    corr, pval = spearmanr(metric1, metric2)

    media1 = np.mean(metric1)
    media2 = np.mean(metric2)
    std1 = np.std(metric1)
    std2 = np.std(metric2)

    print(f"\n📊 {nome_metrica.upper()}")
    print(f"  ➤ Media Dataset 1: {media1:.4f} | STD: {std1:.4f}")
    print(f"  ➤ Media Dataset 2: {media2:.4f} | STD: {std2:.4f}")
    print(f"  ➤ Spearman r = {corr:.2f}, p = {pval:.4f}")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=rank1, y=rank2)
    plt.plot([0, len(rank1)], [0, len(rank1)], '--', color='gray', label='y = x')
    plt.title(f'Confronto ranking {nome_metrica}\nSpearman r = {corr:.2f}, p = {pval:.4f}')
    plt.xlabel(f'Ranking {nome_metrica} Dataset 1')
    plt.ylabel(f'Ranking {nome_metrica} Dataset 2')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def analizza_fd(array1, array2, nomi_file):
    picchi1, medie1, std1 = processa_dataset(array1)
    picchi2, medie2, std2 = processa_dataset(array2)

    visualizza_metriche(picchi1, picchi2, nomi_file, "Picco FD")
    visualizza_metriche(medie1, medie2, nomi_file, "Media FD")
    visualizza_metriche(std1, std2, nomi_file, "Deviazione standard FD")
