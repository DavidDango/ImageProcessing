def test_calc(i, j, B, w, h):
    p = i*B/w
    q = j*B/h
    l_pos = int(p - 0.5)
    n_pos = int(q - 0.5)
    r_pos = int(p + 0.5)
    s_pos = int(q + 0.5)
    dist = p - int(p)
    if dist < 0.5:
        l_weight = 0.5 - dist
        r_weight = 1 - l_weight
    else:
        r_weight = dist - 0.5
        l_weight = 1 - r_weight
    dist = q - int(q)
    if dist < 0.5:
        n_weight = 0.5 - dist
        s_weight = 1 - n_weight
    else:
        s_weight = dist - 0.5
        n_weight = 1 - s_weight
    return [(l_pos, n_pos, l_weight, n_weight), (r_pos, n_pos, r_weight, n_weight), (l_pos, s_pos, l_weight, s_weight), (r_pos, s_pos, r_weight, s_weight)]

print(test_calc(9, 6, 3, 15, 9))