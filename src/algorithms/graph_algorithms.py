#!/usr/bin/env python

from typing import Any, Dict, List, Set, Tuple, Optional, Union, Generator

import heapq

from collections import deque

def HierholzerAlgorithm(edges: List[list], start=None, sort: bool=False, reverse: bool=False) -> List[int]:
    """
    Finding an Eulerian path for a digraph using Hierholzer's algorithm.
    If none exists, returns empty list
    Can be used to solve Leetcode #332, #2097
    """
    out_edges = {}
    degrees = {}
    for e in edges:
        out_edges.setdefault(e[0], [])
        out_edges.setdefault(e[1], [])
        out_edges[e[0]].append(e[1])
        degrees[e[0]] = degrees.get(e[0], 0) + 1
        degrees[e[1]] = degrees.get(e[1], 0) - 1
    if start not in degrees.keys(): return []
    # Checking if Eulerian path exists, and whether (if start is given)
    # and Eulerian path can start at start.
    nonzero_degree = {}
    for v, d in degrees.items():
        if not d: continue
        elif abs(d) != 1 or d in nonzero_degree.keys(): return []
        nonzero_degree[d] = v
    if nonzero_degree:
        if start is not None and nonzero_degree[1] != start: return []
        start = nonzero_degree[1]
    elif start is None:
        start = next(iter(degrees.keys()))
    if sort:
        for v in out_edges.keys():
            out_edges[v].sort(reverse=not reverse)
    
    res = []
    def dfs(v: int) -> List[int]:
        while out_edges[v]:
            v2 = out_edges[v].pop()
            dfs(v2)
        res.append(v)
        return

    dfs(start)
    if len(res) != len(edges) + 1: return []
    return res[::-1]

def findItinerary(tickets: List[List[str]]) -> List[str]:
    """
    Solution to Leetcode #332
    """
    start = "JFK"
    return HierholzerAlgorithm(tickets, start=start, sort=True, reverse=False)

def weightedDirectedEdgesToAdj(v_lst: list, edges: List[list], choice_func=min) -> dict:
    out_edges = {v: {} for v in v_lst}
    for e in edges:
        out_edges[e[0]][e[1]] = choice_func(out_edges[e[0]].get(e[1]), e[2])
    return out_edges

def weightedUndirectedEdgesToAdj(v_lst: list, edges: List[list], choice_func=min) -> dict:
    out_edges = {v: {} for v in v_lst}
    for e in edges:
        out_edges[e[0]][e[1]] = choice_func(out_edges[e[0]].get(e[1]), e[2])
        out_edges[e[1]][e[0]] = out_edges[e[0]][e[1]]
    return out_edges

def DijkstraAdj(out_edges: dict, start) -> dict:
    res = {}
    heap = [(0, start)]
    while heap:
        d, v = heapq.heappop(heap)
        if v in res.keys(): continue
        res[v] = d
        for v2, d2 in out_edges[v].items():
            if v2 in res.keys(): continue
            heapq.heappush(heap, (d + d2, v2))
    return res

def DijkstraEdge(v_lst: list, edges: List[list], start) -> dict:
    return DijkstraAdj(weightedDirectedEdgesToAdj(v_lst, edges), start)

def FloydWarshallAdj(out_adj: dict) -> dict:
    # Need to check
    # For weighted directed graph
    # Returns empty dictionary if negative weight cycle detected
    v_lst = out_adj.keys()
    dists = dict(out_adj)
    for v in v_lst:
        # Checking for negative weight 1-cycle
        if dists[v][v] < 0: return {}
    for u in v_lst:
        for v1 in v_lst:
            if u not in dists[v1].keys(): continue
            d1 = dists[v1][u]
            for v2 in dists[u].keys():
                d = d1 + dists[u][v2]
                if d >= dists[v1][v2]: continue
                # Checking for negative weight cycle
                if v1 == v2 and d < 0: return {}
                dists[v1][v2] = d
    for v in v_lst: dists[v][v] = 0
    return dists

def FloydWarshallEdge(v_lst: List[Any], edges: List[Tuple[Any, Any]], thresh: Union[int, float]=float("inf")) -> dict:
    # Need to check
    # For weighted directed graph
    # Returns empty dict if negative weight cycle detected
    return FloydWarshallAdj(weightedDirectedEdgesToAdj(v_lst, edges, choice_func=min))

def FloydWarshallThreshAdj(adj: dict, thresh: Union[int, float]=float("inf"), edges_under_thresh: bool=False) -> dict:
    # Need to check
    # For weighted undirected graph with no negative edge weights
    v_lst = adj.keys()
    if edges_under_thresh: dists = dict(adj)
    else: dists = {v: {v2: d for v2, d in v2_dict.items() if d <= thresh} for v, v2_dict in adj.items()}
    for u in v_lst:
        for v1 in dists[u].keys():
            d1 = dists[v1][u]
            for v2 in dists[u].keys():
                d = d1 + dists[u][v2]
                if d <= thresh:
                    dists[v1][v2] = min(dists[v1][v2], d)
    for v in v_lst: dists[v][v] = 0
    return dists

def FloydWarshallThreshEdge(v_lst: list, edges: List[list], thresh: Union[int, float]=float("inf")) -> dict:
    # Need to check
    # For weighted undirected graph with no negative edge weights
    adj = {v: {} for v in v_lst}
    for e in edges:
        if e[2] > thresh: continue
        adj[e[0]][e[1]] = min(adj[e[0]].get(e[1], float("inf")), e[2])
        adj[e[1]][e[0]] = adj[e[0]][e[1]]
    return FloydWarshallThreshAdj(adj, thresh, edges_under_thresh=True)


def BellmanFordStepLimitAdj(out_edges: dict, start, mx_step: Union[int, float]=float("inf")) -> dict:
    """
    Bellman-Ford Algorithm with digraph as weighted adjacency list where the
    vertices are exactly the keys of out_edges
    """
    row = {x: float("inf") for x in out_edges.keys()}
    updated = {start}
    row[start] = 0
    for _ in range(mx_step):
        prev = dict(row)
        updated2 = set()
        for v in updated:
            d = prev[v]
            for v2, d2 in out_edges[v].items():
                d3 = d + d2
                if d3 < row[v2]:
                    updated2.add(v2)
                    row[v2] = d3
        if not updated2: break
        updated = updated2
    return {k: (v if isinstance(v, int) else -1) for k, v in row.items()}

def BellmanFordStepLimitEdges(v_lst: list, edges: List[list], start, mx_step: Union[int, float]=float("inf")) -> dict:
    """
    Bellman-Ford Algorithm with digraph as list of weighted directed edges where
    vertices are exactly the contents of v_lst
    """
    return BellmanFordStepLimitAdj(weightedDirectedEdgesToAdj(v_lst, edges), start, mx_step)

def SPFAAdj(out_edges: dict, start: int, weight_comb=lambda x, y: x + y) -> dict:
    """
    Shortest Path Faster Algorithm (SPFA) variant of Bellman-Ford
    Returns empty dict if negative weight cycle is detected
    """
    n = len(out_edges)
    dists = {x: float("inf") for x in out_edges.keys()}
    dists[start] = 0
    qu = deque([(start, 0)])
    in_qu = {start}
    while qu:
        v, n_step = qu.popleft()
        in_qu.remove(v)
        d = dists[v]
        for v2, d2 in out_edges[v].items():
            d3 = weight_comb(d, d2)
            if d3 >= dists[v2]: continue
            if n_step == n - 1: return {}
            dists[v2] = d3
            if v2 in in_qu: continue
            qu.append((v2, n_step + 1))
            in_qu.add(v2)
    return dists

def KahnAdj(out_edges: dict, in_degrees: Optional[dict]=None) -> list:
    """
    Kahn's algorithm for topological sorting with digraph as adjacency list
    where the vertices are exactly the keys of out_edges
    Returns empty list if a cycle is detected
    """
    n = len(out_edges)
    if in_degrees is None:
        in_degrees = {v: 0 for v in out_edges.keys()}
        for v_set in out_edges.values():
            for v in v_set:
                in_degrees[v] += 1
    else: in_degrees = dict(in_degrees) # Ensures argument in_degrees is not altered
    qu = deque()
    for v, in_deg in in_degrees.items():
        if not in_deg: qu.append(v)
    res = []
    while qu:
        v = qu.popleft()
        res.append(v)
        for v2 in out_edges[v]:
            in_degrees[v2] -= 1
            if not in_degrees[v2]:
                qu.append(v2)
    return res if len(res) == n else []


def KahnEdge(v_lst: list, edges: List[list]) -> list:
    """
    Kahn's algorithm for topological sorting with digraph as list of
    directed edges where the vertices are exactly the contents of v_lst
    """
    out_edges = {v: set() for v in v_lst}
    in_degrees = {v: 0 for v in v_lst}
    for e in edges:
        out_edges[e[0]].add(e[1])
        in_degrees[e[1]] += 1
    return KahnAdj(out_edges, in_degrees=in_degrees)

def alienOrder(words: List[str]) -> str:
    """
    Solution to Leetcode #269 (Alien Dictionary) using Kahn's algorithm
    """
    n = len(words)
    out_edges = {}
    for i1, w1 in enumerate(words):
        for l1 in w1: out_edges.setdefault(l1, set())
        for i2 in range(i1 + 1, n):
            w2 = words[i2]
            for l1, l2 in zip(w1, w2):
                if l1 != l2:
                    out_edges[l1].add(l2)
                    break
            else:
                if len(w2) < len(w1): return ""
    res = KahnAdj(out_edges)
    return "".join(res)

def KahnLayeringAdj(out_edges: dict, in_degrees: Optional[dict]=None) -> List[list]:
    """
    Kahn's algorithm for topological sorting with digraph as adjacency list
    where the vertices are exactly the keys of out_edges, partitioning
    the vertices into layers such that each vertex only has
    incoming edges from vertices in earlier layers and each vertex
    is in the earliest possible layer subject to this constraint.
    If cycle is detected, returns an empty list.
    """
    n = len(out_edges)
    if in_degrees is None:
        in_degrees = {v: 0 for v in out_edges.keys()}
        for v_set in out_edges.values():
            for v in v_set:
                in_degrees[v] += 1
    else: in_degrees = dict(in_degrees) # Ensures argument in_degrees is not altered
    qu = deque()
    for v, in_deg in in_degrees.items():
        if not in_deg: qu.append(v)
    res = []
    
    l = len(qu)
    n_seen = l
    for depth in range(1, n + 1):
        res.append([])
        for _ in range(l):
            v = qu.popleft()
            res[-1].append(v)
            for v2 in out_edges[v]:
                in_degrees[v2] -= 1
                if not in_degrees[v2]:
                    qu.append(v2)
        l = len(qu)
        n_seen += l
        if not l or n_seen == n: break
    if l: res.append(list(qu))
    return res if n_seen == n else []


def KahnLayeringEdge(v_lst: list, edges: List[list]) -> List[list]:
    """
    Kahn's algorithm for topological sorting with digraph as list of
    directed edges where the vertices are exactly the contents of v_lst,
    partitioning the vertices into layers such that each vertex only has
    incoming edges from vertices in earlier layers and each vertex
    is in the earliest possible layer subject to this constraint.
    If cycle is detected, returns an empty list.
    """
    out_edges = {v: set() for v in v_lst}
    in_degrees = {v: 0 for v in v_lst}
    for e in edges:
        out_edges[e[0]].add(e[1])
        in_degrees[e[1]] += 1
    print(out_edges, in_degrees)
    return KahnLayeringAdj(out_edges, in_degrees=in_degrees)

def minimumSemesters(n: int, relations: List[List[int]]) -> int:
    """
    Solution to Leetcode #1136 (Parallel Courses) using Kahn's
    algorithm
    """
    if not n: return 0
    res = len(KahnLayeringEdge(list(range(1, n + 1)), relations))
    return res if res else -1

def KosarajuAdj(out_edges: dict) -> Tuple[dict]:
    """
    Kosaraju algorithm for finding strongly connected components (SCC) in a
    directed graph (with the directed graph given as a dictionary,
    where the keys are the vertices with the corresponding value
    being a set containing all the other vertices this vertex has
    a directed edge to- effectively the adjacency list representation
    of the outgoing edges).
    Each SCC is represented by one of its members. Returns a dictionary
    whose keys are the vertices of the original graph with the
    corresponding value being the representative vertex of the SCC to
    which it belongs.
    
    Can be used to solve Leetcode #1557 (when generalised to allow
    directed graphs with cycles- see below), #2101 (see below). In the
    examples included below, when input argument alg given as "Kosaraju",
    KosarajuAdj() applied through condenseSCCAdj().
    """
    in_edges = {v: set() for v in out_edges.keys()}
    for v1, v2_set in out_edges.items():
        for v2 in v2_set:
            in_edges[v2].add(v1)
    seen = set()
    scc_repr = {}
    top_sort_rev = []
    
    def visit(v) -> None:
        if v in seen: return
        seen.add(v)
        for v2 in out_edges[v]:
            visit(v2)
        top_sort_rev.append(v)
        return
        
    def assign(v, v0) -> None:
        if v in scc_repr.keys(): return
        scc_repr[v] = v0
        for v2 in in_edges[v]:
            assign(v2, v0)
        return
    
    for v in out_edges.keys():
        visit(v)
    for v in reversed(top_sort_rev):
        assign(v, v)
    return scc_repr
    
def TarjanSCCAdj(out_edges: dict) -> dict:
    """
    Tarjan algorithm for finding strongly connected components (SCC) in a
    directed graph (with the directed graph given as a dictionary,
    where the keys are the vertices with the corresponding value
    being a set containing all the other vertices this vertex has
    a directed edge to- effectively the adjacency list representation
    of the outgoing edges).
    Each SCC is represented by one of its members. Returns a dictionary
    whose keys are the vertices of the original graph with the
    corresponding value being the representative vertex of the SCC to
    which it belongs.
    
    Can be used to solve Leetcode #1557 (when generalised to allow
    directed graphs with cycles- see below), #2101 (see below) In the
    examples included below, when input argument alg given as "Tarjan",
    TarjanSCCAdj() applied through condenseSCCAdj().
    """
    lo = {}
    stk = []
    in_stk = set()
    scc_repr = {}

    def recur(v: int, t: int) -> int:
        if v in lo.keys():
            return t
        t0 = t
        lo[v] = t
        stk.append(v)
        in_stk.add(v)
        t += 1
        for v2 in out_edges[v]:
            t = recur(v2, t)
            if v2 in in_stk:
                lo[v] = min(lo[v], lo[v2])
        if lo[v] != t0: return t
        while stk[-1] != v:
            v2 = stk.pop()
            in_stk.remove(v2)
            scc_repr[v2] = v
        stk.pop()
        in_stk.remove(v)
        scc_repr[v] = v
        return t
    t = 0
    for v in out_edges.keys():
        t = recur(v, t)
    return scc_repr

def condenseSCCAdj(out_edges: dict, alg: str="Tarjan") -> Tuple[dict]:
    """
    Finds the strongly connected components (SCC) in a
    directed graph (with the directed graph given as a dictionary,
    where the keys are the vertices with the corresponding value
    being a set containing all the other vertices this vertex has
    a directed edge to- effectively the adjacency list representation
    of the outgoing edges) and generates a directed graph where each
    SCC is condensed down to one of its member vertices. Uses either
    Tarjan's algorithm for SCC (if alg given as "Tarjan") or Kosaraju's
    algorithm (if alg given as "Kosaraju").
    
    Each SCC is represented by one of its members. Returns a 4-tuple whose
    0th index is a dictionary whose keys are the vertices of the original
    graph with the corresponding value being the representative vertex
    of the SCC to which it belongs; the 1st index is a dictionary whose
    keys are the vertices chosen to represent each SCC with the corresponding
    value being a set containing all of the vertices in that SCC; the 2nd
    index is a dictionary representing the adjacency list representation
    of the directed graph (similar to the input) with each SCC condensed
    down to its representative member as a single vertex; and 3rd index
    is a dictionary whose keys are the vertices of the condensed directed
    graph of the 2nd index (i.e. the vertices representing each SCC) with
    the corresponding value being the in-degree of this vertex in the
    condensed graph.
    
    Note that the resulting directed graph is guaranteed to be acyclic.
    
    Can be used to solve Leetcode #1557 (when generalised to allow
    directed graphs with cycles- see below), #2101 (see below)
    """
    if alg not in {"Tarjan", "Kosaraju"}:
        raise ValueError("Input argument alg must be either 'Tarjan' or "
            'Kosarau')
    func = TarjanSCCAdj if alg == "Tarjan" else KosarajuAdj
    
    scc_repr = func(out_edges)
    scc_groups = {}
    for v1, v2 in scc_repr.items():
        scc_groups.setdefault(v2, set())
        scc_groups[v2].add(v1)
    out_edges_scc = {v: set() for v in scc_groups.keys()}
    in_degree_scc = {v: 0 for v in scc_groups.keys()}
    
    for v1, v2_set in out_edges.items():
        r1 = scc_repr[v1]
        for v2 in v2_set:
            r2 = scc_repr[v2]
            if r2 == r1: continue
            out_edges_scc[r1].add(r2)
            in_degree_scc[r2] += 1
    return (scc_repr, scc_groups, out_edges_scc, in_degree_scc)
    

def findSmallestSetOfVertices(n: int, edges: List[List[int]], alg: str="Tarjan") -> List[int]:
    """
    Modified Leetcode #1557 (demonstrates possible use of KosarauAdj() and
    TarjanSCCAdj() through condenseSCCAdj())
    
    Original description:
    
    Given a directed acyclic graph, with n vertices numbered from 0 to
    n-1, and an array edges where edges[i] = [fromi, toi] represents a
    directed edge from node fromi to node toi.

    Find the smallest set of vertices from which all nodes in the graph
    are reachable. It's guaranteed that a unique solution exists.

    Notice that you can return the vertices in any order.
    
    Modification: Directed graph is not necessarily acyclic (much harder
    problem)- gives one of the possible solutions
    
    Input argument alg can be "Tarjan" or "Kosaraju", the former using
    TarjanSCCAdj() and the latter using KosarauAdj() (through
    condenseSCCAdj()).
    """
    # Note to get every possible solution, give all combinations where
    # each SCC is represented by each of its members
    out_edges = {v: set() for v in range(n)}
    for e in edges:
        out_edges[e[0]].add(e[1])

    in_degree_scc = condenseSCCAdj(out_edges, alg=alg)[3]
    return [x for x, y in in_degree_scc.items() if not y]

def maximumDetonation(bombs: List[List[int]], alg: str="Tarjan") -> int:
    """
    Leetcode #2101 (demonstrates possible use of KosarauAdj() and
    TarjanSCCAdj() through condenseSCCAdj())
    
    Description:
    
    You are given a list of bombs. The range of a bomb is defined as the
    area where its effect can be felt. This area is in the shape of a
    circle with the center as the location of the bomb.

    The bombs are represented by a 0-indexed 2D integer array bombs where
    bombs[i] = [xi, yi, ri]. xi and yi denote the X-coordinate and
    Y-coordinate of the location of the ith bomb, whereas ri denotes the
    radius of its range.

    You may choose to detonate a single bomb. When a bomb is detonated,
    it will detonate all bombs that lie in its range. These bombs will
    further detonate the bombs that lie in their ranges.

    Given the list of bombs, return the maximum number of bombs that can
    be detonated if you are allowed to detonate only one bomb.
    
    Input argument alg can be "Tarjan" or "Kosaraju", the former using
    TarjanSCCAdj() and the latter using KosarajuAdj() (through
    condenseSCCAdj()).
    """
    n = len(bombs)
    out_edges = {i: set() for i in range(n)}
    dist_sq = [0] * n
    for i1, (x1, y1, r1) in enumerate(bombs):
        dist_sq[i1] = r1 ** 2
        for i2 in range(i1):
            x2, y2, r2 = bombs[i2]
            ds = (x2 - x1) ** 2 + (y2 - y1) ** 2
            if ds <= dist_sq[i1]:
                out_edges[i1].add(i2)
            if ds <= dist_sq[i2]:
                out_edges[i2].add(i1)

    scc_repr, scc_groups, out_edges_scc, in_degree_scc =\
            condenseSCCAdj(out_edges, alg=alg)
    sz_dict = {v: len(v2_set) for v, v2_set in scc_groups.items()}
    memo = {}
    def dfs(v: int) -> Set[int]:
        args = v
        if args in memo.keys(): return memo[args]
        res = {v}
        for v2 in out_edges_scc[v]:
            res |= dfs(v2)
        memo[args] = res
        return res
    
    res = 0
    for v in out_edges_scc.keys():
        if in_degree_scc[v]: continue
        res = max(res, sum(sz_dict[v2] for v2 in dfs(v)))
    return res

def TarjanBridgeAdj(graph: dict) -> Tuple[tuple]:
    """
    Tarjan bridge algorithm. Identifies all bridges in a
    graph. A bridge is an edge of the graph which, if
    removed, increases the number of connected components
    of the graph (or equivalently, causes at least one
    pair of vertices which are connected in the original
    graph to become disconnected).
    
    The graph is given as a dictionary, where the keys are
    the vertices with the corresponding value being a set
    containing all the other vertices with which this vertex
    shares an edge- effectively the adjacency list representation
    of the edges)
    
    Returns a tuple of 2-tuples, with each such tuple representing
    a bridge- the two items in a given 2-tuple are the vertices
    between which the bridge crosses.
    
    Assumes the graph has no repeated edges.
    
    Can be used to solve Leetcode #1192 (see below)
    """
    bridges = []
    lo = {}

    def dfs(v, t: int, parent=None) -> int:
        if v in lo.keys(): return t
        t0 = t
        lo[v] = t
        t += 1
        for v2 in graph[v]:
            if v2 == parent: continue
            t = dfs(v2, t, v)
            lo[v] = min(lo[v], lo[v2])
            if lo[v2] > t0:
                bridges.append((v, v2))
        return t
    
    t = 0
    for v in graph.keys():
        t = dfs(v, t)
    return tuple(bridges)

def criticalConnections(n: int, connections: List[List[int]]) -> List[List[int]]:
    """
    Leetcode #1192 (demonstrates possible use of TarjanBridgeAdj)
    
    Description:
    
    There are n servers numbered from 0 to n - 1 connected by
    undirected server-to-server connections forming a network
    where connections[i] = [ai, bi] represents a connection
    between servers ai and bi. Any server can reach other
    servers directly or indirectly through the network.

    A critical connection is a connection that, if removed,
    will make some servers unable to reach some other server.

    Return all critical connections in the network in any order.
    """
    graph = {v: set() for v in range(n)}
    for e in connections:
        graph[e[0]].add(e[1])
        graph[e[1]].add(e[0])

    return TarjanBridgeAdj(graph)

def TarjanArticulationAdj(graph: dict) -> tuple:
    """
    Tarjan articulation point algorithm. Identifies all
    articulation points in an undirected graph. An
    articulation point is a vertex of the graph which, if
    removed along with all of its associated edges
    increases the number of connected components
    of the graph (or equivalently, causes at least one
    pair of vertices which are connected in the original
    graph to become disconnected).
    
    The graph is given as a dictionary, where the keys are
    the vertices with the corresponding value being a set
    containing all the other vertices with which this vertex
    shares an edge- effectively the adjacency list representation
    of the edges)
    
    Returns a tuple containing precisely the vertices which are
    articulation points.
    """
    # Need to check thoroughly
    # Gives correct answer for:
    #  {0: {1}, 1: {0, 2}, 2: {1}} - answer (1,)
    #  {0: {1,2}, 1: {0, 2}, 2: {0,1}} - answer ()
    #  {0: {1, 2, 3}, 1: {0, 2}, 2: {0, 1}, 3: {0, 4, 5},
    #        4: {3}, 5: {3}} - answer (3, 0)
    
    artic = []
    lo = {}

    def dfs(v, t: int, parent=None) -> int:
        if v in lo.keys(): return t
        t0 = t
        lo[v] = t
        t += 1
        for v2 in graph[v]:
            if v2 == parent: continue
            t = dfs(v2, t, v)
            lo[v] = min(lo[v], lo[v2])
        if t > t0 + 1 and lo[v] == t0:
            artic.append(v)
        return t
    
    t = 0
    for v in graph.keys():
        if v in lo.keys(): continue
        lo[v] = t
        child_count = 0
        for v2 in graph[v]:
            if v2 in lo.keys(): continue
            child_count += 1
            t = dfs(v2, t + 1, parent=v)
        if child_count > 1: artic.append(v)
    return tuple(artic)

def TarjanArticulationAdjGrid(grid: List[List[Any]], open_obj: Any) -> Union[List[Set[Tuple[int]]], Set[Tuple[int]]]:
    # Check and adapt to generic graph (Leetcode #1263)
    """
    Modified Tarjan Algorithm for finding articulation points in
    the grid and for each articulation point identifying which adjacent
    vertices (i.e. positions in the grid) remain connected to each
    other after the removal of that articulation point. Also finds
    which non-wall grid positions are connected to each other.
    """
    shape = (len(grid), len(grid[0]))
    def move(pos: Tuple[int, int]) -> Generator[Tuple[int, int], None, None]:
        if pos[0] > 0:
            pos2 = (pos[0] - 1, pos[1])
            if grid[pos2[0]][pos2[1]] == open_obj: yield pos2
        if pos[0] < shape[0] - 1:
            pos2 = (pos[0] + 1, pos[1])
            if grid[pos2[0]][pos2[1]] == open_obj: yield pos2
        if pos[1] > 0:
            pos2 = (pos[0], pos[1] - 1)
            if grid[pos2[0]][pos2[1]] == open_obj: yield pos2
        if pos[1] < shape[0] - 1:
            pos2 = (pos[0], pos[1] + 1)
            if grid[pos2[0]][pos2[1]] == open_obj: yield pos2

    groups = []
    articulation = {}
    low = {}
    times = {}
    def recur(pos: Tuple[int], parent: Optional[Tuple[int]]=None, t: int=0) -> int:
        if pos in times.keys():
            return times[pos]
        groups[-1].add(pos)
        times[pos] = t
        low[pos] = t
        adj_groups = []
        remain = set(move(pos))
        if parent is not None:
            adj_groups.append({parent})
            remain.remove(parent)
            rm_set = set()
            for pos2 in remain:
                if pos2 in times.keys():
                    adj_groups[0].add(pos2)
                    rm_set.add(pos2)
                    low[pos] = min(low[pos], times[pos2])
            remain -= rm_set
        for pos2 in move(pos):
            if pos2 == parent or pos2 not in remain:
                continue
            l = recur(pos2, parent=pos, t=t + 1)
            low[pos] = min(low[pos], l)
            if l < t: i = 0
            else:
                i = -1
                adj_groups.append(set())
            rm_set = set()
            for pos3 in remain:
                if pos3 in low.keys():
                    adj_groups[i].add(pos3)
                    rm_set.add(pos3)
            remain -= rm_set
        if len(adj_groups) > 1:
            articulation[pos] = adj_groups
        return low[pos]
    
    for i, row in enumerate(grid):
        for j, l in enumerate(row):
            pos = (i, j)
            if pos in low.keys() or grid[i][j] == "#":
                continue
            groups.append(set())
            recur(pos)
    return (groups, articulation)
