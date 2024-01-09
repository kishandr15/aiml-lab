def aStarAlgo(start_node, stop_node, graph_nodes):
    open_set = set([start_node])
    closed_set = set()
    g = {start_node: 0}
    parents = {start_node: start_node}

    while len(open_set) > 0:
        n = None
        for v in open_set:
            if n is None or g[v] + heuristic(v) < g[n] + heuristic(n):
                n = v

        if n is None:
            print('Path does not exist!')
            return None

        if n == stop_node:
            path = []
            while parents[n] != n:
                path.append(n)
                n = parents[n]
            path.append(start_node)
            path.reverse()
            print('Path found: {}'.format(path))
            return path

        open_set.remove(n)
        closed_set.add(n)

        if n in graph_nodes:
            for (m, weight) in get_neighbours(n, graph_nodes):
                if m not in open_set and m not in closed_set:
                    open_set.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n
                    if m in closed_set:
                        closed_set.remove(m)
                        open_set.add(m)

    print('Path does not exist!')
    return None


def get_neighbours(v, graph_nodes):
    if v in graph_nodes:
        return graph_nodes[v]
    else:
        return None


def heuristic(n):
    h_dist = {
        'A': 10,
        'B': 7,
        'C': 8,
        'D': 3,
        'E': 6,
        'F': 2,
        'G': 5,
        'H': 1
    }
    return h_dist[n]


graph_nodes = {
    'A': [('B', 5), ('E', 3)],
    'B': [('C', 2), ('D', 2)],
    'E': [('F', 6), ('G', 4)],
    'F': [('H', 9)],
    'G': [('H', 8)]
}

aStarAlgo('A', 'H', graph_nodes)




# OUTPUT:

# Path found: ['A', 'F', 'G', 'I', 'J']