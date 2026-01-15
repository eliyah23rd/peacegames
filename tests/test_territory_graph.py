import unittest

from peacegame.territory_graph import (
    assign_territories_round_robin,
    average_degree,
    build_territory_graph,
    compute_legal_cession_lists,
    load_fixed_map,
)


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

    def test_assignments_and_legal_cessions(self) -> None:
        names = ["Auron", "Bastion", "Caldera", "Driftwood", "Everspring"]
        agents = ["Alpha", "Beta"]
        graph, positions = build_territory_graph(names, seed=7)
        assigned = assign_territories_round_robin(agents, graph, positions, seed=7)

        all_assigned = set().union(*assigned.values())
        self.assertEqual(all_assigned, set(names))
        for agent in agents:
            self.assertGreaterEqual(len(assigned[agent]), 1)

        inbound, outbound = compute_legal_cession_lists(assigned, graph)
        self.assertEqual(set(inbound.keys()), set(agents))
        self.assertEqual(set(outbound.keys()), set(agents))

    def test_capital_not_legal_for_cession(self) -> None:
        names = ["Auron", "Bastion", "Caldera", "Driftwood", "Everspring"]
        agents = ["Alpha", "Beta"]
        graph, positions = build_territory_graph(names, seed=9)
        assigned, capitals = assign_territories_round_robin(
            agents, graph, positions, seed=9, return_capitals=True
        )
        inbound, outbound = compute_legal_cession_lists(
            assigned, graph, capitals=capitals
        )
        for agent, capital in capitals.items():
            if not capital:
                continue
            for receiver, terrs in outbound.get(agent, {}).items():
                self.assertNotIn(capital, terrs)

    def test_fixed_map_assignments_deterministic(self) -> None:
        names, graph, positions = load_fixed_map()
        agents = ["Alpha", "Beta", "Gamma", "Delta"]
        targets = {agent: 8 for agent in agents}
        assigned_a, capitals_a = assign_territories_round_robin(
            agents,
            graph,
            positions,
            seed=17,
            return_capitals=True,
            target_counts=targets,
        )
        assigned_b, capitals_b = assign_territories_round_robin(
            agents,
            graph,
            positions,
            seed=17,
            return_capitals=True,
            target_counts=targets,
        )
        self.assertEqual(assigned_a, assigned_b)
        self.assertEqual(capitals_a, capitals_b)
        self.assertEqual(set(graph.keys()), set(names))
        self.assertEqual(sum(len(v) for v in assigned_a.values()), 32)
        for agent in agents:
            self.assertEqual(len(assigned_a[agent]), 8)


if __name__ == "__main__":
    unittest.main()
