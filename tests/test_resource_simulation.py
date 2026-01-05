import json
import random
import unittest

from peacegame.resource_simulation import (
    RESOURCE_TYPES,
    ResourceSimulationEngine,
    _resource_multiplier,
    _resource_ratios,
    generate_territory_resources,
)


class ScriptedAgent:
    def __init__(self, actions_by_turn: dict[int, dict]) -> None:
        self._actions_by_turn = actions_by_turn

    def act(self, agent_input: dict) -> str:
        turn = int(agent_input.get("turn", 0))
        action = self._actions_by_turn.get(turn, {})
        return json.dumps(action)


class ResourceSimulationTests(unittest.TestCase):
    def test_generate_territory_resources_ranges(self) -> None:
        terrs = ["T1", "T2", "T3", "T4"]
        graph = {t: set() for t in terrs}
        graph["T1"].add("T2")
        graph["T2"].update({"T1", "T3"})
        graph["T3"].update({"T2", "T4"})
        graph["T4"].add("T3")
        resources = generate_territory_resources(
            terrs,
            graph,
            peaks_per_resource={"energy": 1, "minerals": 1, "food": 1},
            max_value=3,
            seed=1,
        )
        self.assertEqual(set(resources.keys()), set(terrs))
        for terr, res in resources.items():
            for rtype, qty in res.items():
                self.assertIn(rtype, RESOURCE_TYPES)
                self.assertTrue(0 < qty <= 3)

    def test_generate_territory_resources_peak_decay(self) -> None:
        terrs = ["A", "B", "C", "D"]
        graph = {"A": {"B"}, "B": {"A", "C"}, "C": {"B", "D"}, "D": {"C"}}
        seed = 5
        rng = random.Random(seed)
        peak = rng.sample(terrs, 1)[0]
        dist = {"A": 0, "B": 1, "C": 2, "D": 3}
        # distance from peak along line
        if peak == "B":
            dist = {"A": 1, "B": 0, "C": 1, "D": 2}
        elif peak == "C":
            dist = {"A": 2, "B": 1, "C": 0, "D": 1}
        elif peak == "D":
            dist = {"A": 3, "B": 2, "C": 1, "D": 0}
        resources = generate_territory_resources(
            terrs,
            graph,
            peaks_per_resource={"energy": 1},
            max_value=3,
            seed=seed,
        )
        for terr in terrs:
            expected = max(0, 3 - dist[terr])
            actual = resources[terr].get("energy", 0)
            self.assertEqual(actual, expected)

    def test_resource_ratios_and_multiplier(self) -> None:
        totals = {
            "A": {"energy": 2, "minerals": 1, "food": 3},
            "B": {"energy": 1, "minerals": 1, "food": 1},
        }
        territories = {"A": {"T1", "T2"}, "B": {"T3"}}
        constants = {"c_min_energy": 1, "c_min_minerals": 1, "c_min_food": 2}
        ratios = _resource_ratios(totals, territories, constants)
        self.assertAlmostEqual(ratios["A"]["energy"], 1.0)
        self.assertAlmostEqual(ratios["A"]["minerals"], 0.5)
        self.assertAlmostEqual(ratios["A"]["food"], 0.75)
        mult = _resource_multiplier(ratios)
        self.assertAlmostEqual(mult["A"], 1.0 * 0.5 * 0.75)

    def test_money_grants_apply_next_turn(self) -> None:
        engine = ResourceSimulationEngine(run_label="resource_test")
        constants = {
            "c_min_energy": 1,
            "c_min_minerals": 1,
            "c_min_food": 1,
            "c_money_per_territory": 10,
            "c_damage_per_attack_mil": 1,
            "c_mil_upkeep_price": 1,
            "c_mil_purchase_price": 100,
            "c_defense_destroy_factor": 2,
            "c_trade_factor": 2.0,
        }
        engine.setup_state(
            agent_territories={"A": {"T1"}, "B": {"T2"}},
            agent_mils={"A": 0, "B": 0},
            agent_welfare={"A": 0, "B": 0},
            territory_seed=1,
            resource_seed=1,
            use_generated_territories=False,
            resource_peaks={"energy": 1, "minerals": 1, "food": 1},
            resource_peak_max=3,
        )
        engine.territory_resources = {
            "T1": {"energy": 1, "minerals": 1, "food": 1},
            "T2": {"energy": 1, "minerals": 1, "food": 1},
        }
        engine.setup_round(total_turns=2)

        agents = {
            "A": ScriptedAgent(
                {
                    0: {"money_grants": {"B": 5}},
                    1: {},
                }
            ),
            "B": ScriptedAgent({0: {}, 1: {}}),
        }

        summaries = {"A": "", "B": ""}
        results_turn0 = engine.run_turn(
            script_name="resource_test",
            turn=0,
            agents=agents,
            constants=constants,
            turn_summaries=summaries,
            max_summary_len=256,
        )
        results_turn1 = engine.run_turn(
            script_name="resource_test",
            turn=1,
            agents=agents,
            constants=constants,
            turn_summaries=results_turn0.get("d_summary_last_turn", summaries),
            max_summary_len=256,
        )
        welfare_turn0 = results_turn0["d_total_welfare_this_turn"]["B"]
        welfare_turn1 = results_turn1["d_total_welfare_this_turn"]["B"]
        self.assertEqual(welfare_turn0, 10)
        self.assertEqual(welfare_turn1, 20)
        engine.close()

    def test_resource_grants_improve_next_turn_income(self) -> None:
        engine = ResourceSimulationEngine(run_label="resource_grant_test")
        constants = {
            "c_min_energy": 2,
            "c_min_minerals": 1,
            "c_min_food": 1,
            "c_money_per_territory": 10,
            "c_damage_per_attack_mil": 1,
            "c_mil_upkeep_price": 1,
            "c_mil_purchase_price": 100,
            "c_defense_destroy_factor": 2,
            "c_trade_factor": 1.0,
        }
        engine.setup_state(
            agent_territories={"A": {"T1"}, "B": {"T2"}},
            agent_mils={"A": 0, "B": 0},
            agent_welfare={"A": 0, "B": 0},
            territory_seed=2,
            resource_seed=2,
            use_generated_territories=False,
            resource_peaks={"energy": 1, "minerals": 1, "food": 1},
            resource_peak_max=3,
        )
        engine.territory_resources = {
            "T1": {"energy": 1, "minerals": 1, "food": 1},
            "T2": {"energy": 1, "minerals": 1, "food": 1},
        }
        engine.setup_round(total_turns=2)

        agents = {
            "A": ScriptedAgent(
                {
                    0: {"resource_grants": {"B": {"energy": 1}}},
                    1: {},
                }
            ),
            "B": ScriptedAgent({0: {}, 1: {}}),
        }

        summaries = {"A": "", "B": ""}
        results_turn0 = engine.run_turn(
            script_name="resource_grant_test",
            turn=0,
            agents=agents,
            constants=constants,
            turn_summaries=summaries,
            max_summary_len=256,
        )
        results_turn1 = engine.run_turn(
            script_name="resource_grant_test",
            turn=1,
            agents=agents,
            constants=constants,
            turn_summaries=results_turn0.get("d_summary_last_turn", summaries),
            max_summary_len=256,
        )
        welfare_turn0 = results_turn0["d_total_welfare_this_turn"]["B"]
        welfare_turn1 = results_turn1["d_total_welfare_this_turn"]["B"]
        self.assertEqual(welfare_turn0, 5)
        self.assertEqual(welfare_turn1, 10)
        engine.close()


if __name__ == "__main__":
    unittest.main()
