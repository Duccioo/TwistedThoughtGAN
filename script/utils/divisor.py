# piccola funzione per ottenere la grandezza del batch 
# in modo che sia divisibile con il totale delle immagini


def getDivisors(n, res=None):
    res = res or []
    i = 1
    while i <= n:
        if n % i == 0:
            res.append(i),
        i = i + 1
    return res


def get_closest_batch_size(n, close_to=1440):
    all_divisors = getDivisors(n)
    for ix, val in enumerate(all_divisors):
        if close_to < val:
            if ix == 0:
                return val
            if (val - close_to) > (close_to - all_divisors[ix - 1]):
                return all_divisors[ix - 1]
            return val
