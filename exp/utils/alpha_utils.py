import numpy as np

def add_to_bigraph(tree, i, j):
    if i in tree:
        tree[i].append(j)
    else:
        tree[i] = [j]
    if j in tree:
        tree[j].append(i)
    else:
        tree[j] = [i]

def get_mst(dist_mat):
    tree = {0: []}
    s = 0
    n = dist_mat.shape[0]
    from_arr = np.full(n, s)
    dist_arr = dist_mat[s]
    for _ in range(n - 1):
        dist_indice = np.argsort(dist_arr)
        for dist_index in dist_indice:
            if dist_index in tree:
                continue
            add_to_bigraph(tree, from_arr[dist_index], dist_index)
            replace_mask = dist_mat[dist_index] < dist_arr
            dist_arr = np.where(replace_mask, dist_mat[dist_index], dist_arr)
            from_arr = np.where(replace_mask, dist_index, from_arr)
            break
    return tree

def get_one_tree(dist_mat):
    mst = get_mst(dist_mat)
    leaves, stalks = zip(*map(lambda pair: (pair[0], pair[1][0]), filter(lambda pair: len(pair[1]) == 1, mst.items())))
    leaves_second_lengths = []
    for leaf, stalk, leaf_dists in zip(leaves, stalks, dist_mat[list(leaves)]):
        arg_sorted_leaf_dists = sorted(enumerate(leaf_dists), key=lambda pair: pair[1])
        for index, dist in arg_sorted_leaf_dists:
            if index != leaf and index != stalk:
                leaves_second_lengths.append((leaf, index, dist))
                break
    leaves_second_lengths.sort(key=lambda triple: triple[2])
    one_node, one_target, _ = leaves_second_lengths[0]
    add_to_bigraph(mst, one_node, one_target)
    return one_node, one_target, mst

def get_alpha(dist_mat, count):
    one_node, one_target, one_tree = get_one_tree(dist_mat)
    longer_length = dist_mat[one_node][one_target]
    n = dist_mat.shape[0]

    alpha_list = []
    alpha_index_list = []
    beta = np.zeros(n)
    for i in range(n):
        beta.fill(0)
        search_list = [i]
        visited = set(search_list)
        connected = set(one_tree[i])
        while search_list:
            current = search_list.pop(0)
            for next_node in one_tree[current]:
                if next_node in visited:
                    continue
                if next_node in connected:
                    beta[next_node] = dist_mat[i][next_node]
                elif i == one_node or current == one_node:
                    beta[next_node] = longer_length
                else:
                    beta[next_node] = max(beta[current], dist_mat[current][next_node])
                visited.add(next_node)
                search_list.append(next_node)
        alpha = dist_mat[i] - beta
        alpha_index = np.argsort(alpha)[:count]
        alpha_list.append(alpha[alpha_index])
        alpha_index_list.append(alpha_index)
    return np.stack(alpha_list), np.stack(alpha_index_list)
