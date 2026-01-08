# utils.py
# -----------------------------------------------------------------------------
# Utilità per:
# 1) costruire .npz da cartelle o liste di .txt (uno per paziente);
# 2) verificare e allineare i pazienti tra due dataset (A/B);
# 3) caricare coppie di dataset .npz e produrre split train/val/test
#    per i 3 scenari richiesti, senza mischiare A e B.
#    - case=1: train A, val/test B sugli stessi pazienti del train
#    - case=2: train A, val/test A su pazienti diversi
#    - case=3: train A, val/test B su pazienti diversi
#
# Assunzioni .txt:
# - ogni file .txt = un paziente; righe = time-step; colonne = feature
# - si VOGLIONO tenere le prime 6 colonne e SCARTARE le successive (di default)
#
# Assunzioni .npz:
# - chiavi: 'data' (shape [N, T, D]) e 'patient_ids' (shape [N])
# - un campione per paziente (se hai più file per paziente, aggrega prima)
# -----------------------------------------------------------------------------

import os
import glob
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ---------------------------- #
# Helpers base
# ---------------------------- #

def _as_float_array(a: np.ndarray) -> np.ndarray:
    """Cast sicuro a np.float32."""
    a = np.asarray(a)
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    return a

def _align_time_dim_pair(A: np.ndarray, B: np.ndarray, mode: Optional[str] = "min") -> Tuple[np.ndarray, np.ndarray]:
    """
    Allinea la dimensione temporale (T) tra A e B.
    mode="min": taglia entrambi alla T minima (semplice e sicuro).
    mode=None: non fa nulla (richiede che T coincida già).
    """
    if mode is None:
        return A, B
    TA, TB = A.shape[1], B.shape[1]
    T = min(TA, TB)
    if TA != T:
        A = A[:, :T, :]
    if TB != T:
        B = B[:, :T, :]
    return A, B

def _align_time_dim_list(arrays: List[np.ndarray], mode: str = "min") -> List[np.ndarray]:
    """
    Allinea la dimensione temporale (T) su una lista di array [T, D].
    mode="min": taglia ogni serie alla T minima.
    mode="pad": pad con zeri fino alla T massima.
    mode=None: richiede T uguale per tutti.
    """
    if mode is None:
        Ts = {a.shape[0] for a in arrays}
        if len(Ts) != 1:
            raise ValueError("Serie con T diverse: imposta align_T='min' o 'pad'.")
        return arrays

    Ts = [a.shape[0] for a in arrays]
    if mode == "min":
        T = min(Ts)
        return [a[:T] for a in arrays]
    elif mode == "pad":
        T = max(Ts)
        D = arrays[0].shape[1]
        out = []
        for a in arrays:
            if a.shape[0] < T:
                pad = np.zeros((T - a.shape[0], D), dtype=np.float32)
                a = np.vstack([a, pad])
            out.append(a)
        return out
    else:
        raise ValueError("align_T deve essere 'min', 'pad' o None.")

def _fit_transform_by_feature(train: np.ndarray, *others: np.ndarray):
    """
    Fit StandardScaler per feature (D) su TRAIN [N, T, D] e trasforma train/others.
    Ritorna (scalers, train_tf, *others_tf) dove scalers è una lista di D scaler.
    """
    N, T, D = train.shape
    scalers = [StandardScaler() for _ in range(D)]
    train_tf = train.copy()
    others_tf = [x.copy() for x in others]

    for d in range(D):
        v_train = train[:, :, d].reshape(-1, 1)
        scalers[d].fit(v_train)
        train_tf[:, :, d] = scalers[d].transform(v_train).reshape(N, T)

        for i, arr in enumerate(others_tf):
            Nd, Td, _ = arr.shape
            v = arr[:, :, d].reshape(-1, 1)
            others_tf[i][:, :, d] = scalers[d].transform(v).reshape(Nd, Td)

    return (scalers, train_tf, *others_tf)

def _basename_wo_ext(path: str) -> str:
    """Nome file senza estensione."""
    return os.path.splitext(os.path.basename(path))[0]


# ---------------------------- #
# Costruzione .npz dai .txt
# ---------------------------- #

def load_txt_matrix(txt_path: str, keep_first_k: int = None, drop_first_k: int = None, dtype=np.float32) -> np.ndarray:
    """
    Legge un .txt come matrice [T, D_raw].
    - Se keep_first_k è valorizzato  -> tiene SOLO le prime k colonne.
    - Altrimenti se drop_first_k > 0 -> scarta le prime k colonne.
    - Se nessuno dei due è dato      -> tiene tutte le colonne.
    (Gestisce anche il caso T=1.)
    """
    arr = np.loadtxt(txt_path, dtype=dtype)
    if arr.ndim == 1:
        arr = arr[None, :]

    if keep_first_k is not None:
        if arr.shape[1] < keep_first_k:
            raise ValueError(f"{txt_path}: colonne={arr.shape[1]} < keep_first_k={keep_first_k}")
        arr = arr[:, :keep_first_k]
    elif (drop_first_k is not None) and (drop_first_k > 0):
        if arr.shape[1] <= drop_first_k:
            raise ValueError(f"{txt_path}: colonne={arr.shape[1]} <= drop_first_k={drop_first_k}")
        arr = arr[:, drop_first_k:]

    return _as_float_array(arr)


def match_patient_txt_files(folderA: str, folderB: str, pattern: str = "*.txt") -> Tuple[List[str], List[str], List[str]]:
    """
    Ritorna tre liste:
      - files_A_aligned: percorsi in A ordinati e filtrati sui pazienti in comune con B
      - files_B_aligned: percorsi in B nello stesso ordine/paziente di sopra
      - patient_ids:     ID (basename senza estensione) allineati 1:1
    Solo pazienti in comune vengono inclusi; le liste sono ordinate per ID.
    """
    filesA = sorted(glob.glob(os.path.join(folderA, pattern)))
    filesB = sorted(glob.glob(os.path.join(folderB, pattern)))
    mapA = {_basename_wo_ext(p): p for p in filesA}
    mapB = {_basename_wo_ext(p): p for p in filesB}

    common_ids = sorted(set(mapA.keys()).intersection(mapB.keys()))
    if not common_ids:
        raise ValueError("Nessun paziente in comune tra le due cartelle.")

    files_A_aligned = [mapA[p] for p in common_ids]
    files_B_aligned = [mapB[p] for p in common_ids]
    return files_A_aligned, files_B_aligned, common_ids

def make_npz_from_txt_files(
    file_list: List[str],
    out_path: str,
    keep_first_k: int = 6,            # default: tieni le prime 6 colonne
    drop_first_k: int = None,         # opzionale, solo per retrocompatibilità
    align_T: Optional[str] = "min",   # "min" | "pad" | None
) -> Tuple[str, Tuple[int, int, int], np.ndarray]:
    """
    Converte una lista ordinata di .txt in un .npz con:
      - 'data' [N, T, D]
      - 'patient_ids' [N]
    """
    arrays, ids = [], []
    for p in file_list:
        arr = load_txt_matrix(p, keep_first_k=keep_first_k, drop_first_k=drop_first_k)
        arrays.append(arr)
        ids.append(_basename_wo_ext(p))

    arrays = _align_time_dim_list(arrays, mode=align_T)
    data = np.stack(arrays, axis=0)  # [N, T, D]
    patient_ids = np.asarray(ids)

    np.savez_compressed(out_path, data=data, patient_ids=patient_ids)
    return out_path, data.shape, patient_ids


def make_npz_from_txt_folder(
    folder_path: str,
    out_path: str,
    keep_first_k,            # default: tieni le prime 6 colonne
    drop_first_k: int = None,         # opzionale, solo per retrocompatibilità
    align_T: Optional[str] = "min",
    sort: bool = True,
) -> Tuple[str, Tuple[int, int, int], np.ndarray]:
    """
    Legge tutti i .txt nella cartella e costruisce il .npz.
    """
    files = glob.glob(os.path.join(folder_path, "*.txt"))
    if not files:
        raise ValueError(f"Nessun .txt trovato in {folder_path}")
    if sort:
        files = sorted(files)
    return make_npz_from_txt_files(
        files, out_path,
        keep_first_k=keep_first_k,
        drop_first_k=drop_first_k,
        align_T=align_T
    )



# ---------------------------- #
# Split per i 3 scenari 
# ---------------------------- #

def load_data_by_patient(
    dataA_path: str,
    dataB_path: str,
    case: int = 1,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    enforce_same_order: bool = True,
    align_T_mode: Optional[str] = "min",
):
    """
    Carica due dataset .npz (A/B) e produce split train/val/test per i tre casi:

    case=1: Train su A (subset pazienti), Val/Test su B sugli *stessi pazienti*.
    case=2: Train su A (subset pazienti), Val/Test su A su *pazienti diversi*.
    case=3: Train su A (subset pazienti), Val/Test su B su *pazienti diversi*.

    .npz attesi con chiavi:
      - 'data' [N, T, D]
      - 'patient_ids' [N]

    Returns:
      scalers, X_train, X_val, X_test, train_ids, test_ids
      (scalers = lista StandardScaler per D feature, fittati su TRAIN)
    """
    # ---- load ----
    dsA = np.load(dataA_path, allow_pickle=True)
    dsB = np.load(dataB_path, allow_pickle=True)

    dataA = _as_float_array(dsA["data"])
    dataB = _as_float_array(dsB["data"])
    patientsA = np.asarray(dsA["patient_ids"])
    patientsB = np.asarray(dsB["patient_ids"])

    # ---- controlli / allineamenti (ordine pazienti) ----
    if enforce_same_order:
        if len(patientsA) != len(patientsB):
            raise ValueError("Lunghezza patient_ids diversa tra A e B.")
        if not np.array_equal(patientsA, patientsB):
            raise AssertionError(
                "I due dataset devono avere stessi pazienti nello stesso ordine "
                "(oppure usa enforce_same_order=False)."
            )
    else:
        # Riallinea A/B sull'intersezione, seguendo l'ordine di A
        posB = {p: i for i, p in enumerate(patientsB)}
        common = [p for p in patientsA if p in posB]
        if not common:
            raise ValueError("Nessun paziente in comune tra A e B.")
        idxA = np.array([i for i, p in enumerate(patientsA) if p in posB], dtype=int)
        idxB = np.array([posB[p] for p in common], dtype=int)
        dataA, patientsA = dataA[idxA], np.asarray(common)
        dataB, patientsB = dataB[idxB], np.asarray(common)

    # ---- allinea T tra A e B se necessario ----
    dataA, dataB = _align_time_dim_pair(dataA, dataB, mode=align_T_mode)

    # ---- split per i casi ----
    if case == 1:
        # Train su A (train_ids), Val/Test su B sugli *stessi* pazienti
        train_ids, _ = train_test_split(
            patientsA, test_size=test_size, random_state=random_state, shuffle=True
        )
        mA_train = np.isin(patientsA, train_ids)
        mB_same  = np.isin(patientsB, train_ids)

        X_train = dataA[mA_train]      # TRAIN da A
        X_evalB = dataB[mB_same]       # Val/Test da B (stessi pazienti)

        # split 50/50 by-sample dentro B
        if len(X_evalB) >= 2:
            X_val, X_test = train_test_split(
                X_evalB, test_size=0.5, random_state=random_state, shuffle=True
            )
        else:
            X_val = X_evalB[:0]
            X_test = X_evalB

        test_ids = train_ids.copy()    # semantica: stessi pazienti del train

    elif case == 2:
        # Train su A (train_ids), Val/Test su A (test_ids), pazienti disgiunti
        train_ids, test_ids = train_test_split(
            patientsA, test_size=test_size, random_state=random_state, shuffle=True
        )
        mA_train = np.isin(patientsA, train_ids)
        mA_test  = np.isin(patientsA, test_ids)

        X_train = dataA[mA_train]
        X_test  = dataA[mA_test]

        # Val by-sample dal TRAIN
        if len(X_train) >= 2 and val_size > 0:
            X_train, X_val = train_test_split(
                X_train, test_size=val_size, random_state=random_state, shuffle=True
            )
        else:
            X_val = X_train[:0]

    elif case == 3:
        # Train su A (train_ids), Val/Test su B (test_ids), pazienti disgiunti
        train_ids, test_ids = train_test_split(
            patientsA, test_size=test_size, random_state=random_state, shuffle=True
        )
        mA_train = np.isin(patientsA, train_ids)
        mB_test  = np.isin(patientsB, test_ids)

        X_train = dataA[mA_train]
        X_test  = dataB[mB_test]

        # Val by-sample dal TRAIN (su A)
        if len(X_train) >= 2 and val_size > 0:
            X_train, X_val = train_test_split(
                X_train, test_size=val_size, random_state=random_state, shuffle=True
            )
        else:
            X_val = X_train[:0]
    else:
        raise ValueError("case deve essere 1, 2, oppure 3.")

    # ---- normalizzazione: fit SOLO su TRAIN ----
    scalers, X_train, X_val, X_test = _fit_transform_by_feature(X_train, X_val, X_test)

    return scalers, X_train, X_val, X_test, train_ids, test_ids


def load_data(
    dataA_path: str,
    dataB_path: Optional[str] = None,
    case: int = 2,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    enforce_same_order: bool = True,
    align_T_mode: Optional[str] = "min",
):
    """
    Wrapper compatibile con notebook esistenti.
    Se dataB_path è None, usa dataA_path anche come B (utile per case=2).
    """
    if dataB_path is None:
        dataB_path = dataA_path
    return load_data_by_patient(
        dataA_path=dataA_path,
        dataB_path=dataB_path,
        case=case,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        enforce_same_order=enforce_same_order,
        align_T_mode=align_T_mode,
    )


# ---------------------------- #
# Salvataggi split (comodo per notebook)
# ---------------------------- #

def save_split_npz(
    out_path: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    scalers: Optional[List[StandardScaler]] = None,
):
    """
    Salva train/val/test e (opz.) parametri degli scaler in un singolo .npz.
    """
    to_save = dict(
        X_train=X_train, X_val=X_val, X_test=X_test,
        train_ids=np.asarray(train_ids), test_ids=np.asarray(test_ids)
    )
    if scalers is not None and len(scalers) > 0:
        means = np.array([scaler.mean_[0] for scaler in scalers], dtype=np.float32)
        stds  = np.array([scaler.scale_[0] for scaler in scalers], dtype=np.float32)
        to_save["scaler_means"] = means
        to_save["scaler_stds"]  = stds
    np.savez_compressed(out_path, **to_save)
    return out_path
