import unittest

from peacegame.territory_graph import average_degree, build_territory_graph


def _connected(graph: dict[str, set[str]]) -> bool:
    if not graph:
        return True
    start = next(iter(graph))
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor in graph[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == len(graph)


class TerritoryGraphTests(unittest.TestCase):
    def test_graph_connected_and_bounded_degree(self) -> None:
        names = [f"Terr{i}" for i in range(20)]
        graph, positions = build_territory_graph(names, seed=42)

        self.assertEqual(set(graph.keys()), set(names))
        self.assertEqual(set(positions.keys()), set(names))
        self.assertTrue(_connected(graph))

        for neighbors in graph.values():
            self.assertLessEqual(len(neighbors), 4)

        avg = average_degree(graph)
        self.assertGreaterEqual(avg, 2.0)
        self.assertLessEqual(avg, 3.2)


if __name__ == "__main__":
    unittest.main()
