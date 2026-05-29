import numpy as np

def summarize_intervals(W, payments, gains, N, m):


    W = np.array(W)
    payments = np.array(payments)
    gains = np.array(gains)

    # =========================
    # 1. cut
    # =========================
    interval_size = int(np.ceil(N / m))

    interval_winner_counts = np.zeros(m, dtype=int)
    interval_payment_sums = np.zeros(m)
    

    # =========================
    # 2. intervals
    # =========================
    for w, p in zip(W, payments):
        interval_idx = min(w // interval_size, m - 1)

        interval_winner_counts[interval_idx] += 1
        interval_payment_sums[interval_idx] += p

    # =========================
    # 3. cumulative gain
    # =========================
    gain_array = np.zeros(N)
    payments_array = np.zeros(N)

    for w, g in zip(W, gains):
        gain_array[w] += g

    for w, p in zip(W, payments):
        payments_array[w] += p 


    cumulative_gain = np.cumsum(gain_array)
    cumulative_payment = np.cumsum(payments_array)

    return interval_winner_counts, cumulative_payment, cumulative_gain
