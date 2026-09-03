from run_segment_pricing_mdp import solve_budget_mdp


def action(name, cost, reward):
    return {
        "action_id": name, "daily_expected_markdown_spend": cost,
        "daily_incremental_list_value_mean": reward,
        "daily_incremental_list_value_lcb95": reward,
        "daily_incremental_post_discount_sales": reward - cost,
        "daily_incremental_distinct_products": reward / 10,
    }


def test_budget_mdp_uses_budget_on_highest_value_feasible_sequence():
    answer = solve_budget_mdp(
        [action("none", 0, 0), action("small", 2, 3), action("large", 5, 9)],
        horizon=3, budget=10, bins=100, utilization_floor=0.95)
    assert answer["feasible"]
    assert answer["action_day_counts"] == {"none": 1, "large": 2}
    assert answer["total_incremental_list_value_mean"] == 18


def test_budget_mdp_reports_infeasible_mandatory_spend():
    answer = solve_budget_mdp(
        [action("none", 0, 0), action("small", 1, 2)],
        horizon=2, budget=10, bins=100, utilization_floor=0.95)
    assert not answer["feasible"]
