import pytest

from run_segment_pricing_mdp import objective_reward, solve_budget_mdp


def action(name, cost, reward, net=None):
    net = reward - cost if net is None else net
    return {
        "action_id": name, "daily_expected_markdown_spend": cost,
        "daily_reward_mean": reward, "daily_reward_lcb95": reward,
        "daily_incremental_list_value_mean": reward,
        "daily_incremental_list_value_lcb95": reward,
        "daily_incremental_post_discount_sales": net,
        "daily_incremental_distinct_products": reward / 10,
    }


def net_sales_action(name, cost, list_value):
    """An action scored by post-discount sales: list value minus the markdown it pays."""
    row = action(name, cost, list_value)
    row["daily_reward_mean"] = row["daily_reward_lcb95"] = list_value - cost
    return row


def test_budget_mdp_uses_budget_on_highest_value_feasible_sequence():
    answer = solve_budget_mdp(
        [action("none", 0, 0), action("small", 2, 3), action("large", 5, 9)],
        horizon=3, budget=10, bins=100, utilization_floor=0.95)
    assert answer["feasible"]
    assert answer["action_day_counts"] == {"none": 1, "large": 2}
    assert answer["total_incremental_list_value_mean"] == 18
    assert answer["total_reward_mean"] == 18


def test_budget_mdp_reports_infeasible_mandatory_spend():
    answer = solve_budget_mdp(
        [action("none", 0, 0), action("small", 1, 2)],
        horizon=2, budget=10, bins=100, utilization_floor=0.95)
    assert not answer["feasible"]


def test_net_sales_objective_rejects_promotion_that_only_offsets_its_markdown():
    # Raises shelf-price value by 4/day but pays 5/day in markdown: list-value
    # optimization runs it every day, net-sales optimization never does.
    actions_list = [action("none", 0, 0), action("offset", 5, 4)]
    actions_net = [net_sales_action("none", 0, 0), net_sales_action("offset", 5, 4)]
    by_list = solve_budget_mdp(actions_list, 3, 15, 150, 0)
    by_net = solve_budget_mdp(actions_net, 3, 15, 150, 0)
    assert by_list["action_day_counts"] == {"offset": 3}
    assert by_list["total_incremental_post_discount_sales"] == -3
    assert by_net["action_day_counts"] == {"none": 3}
    assert by_net["total_reward_mean"] == 0


def test_objective_reward_selects_declared_quantity_and_its_lower_bound():
    row = {"incremental_list_value": 4.0, "incremental_list_value_lcb95": 3.0,
           "incremental_post_discount_sales": -1.0,
           "incremental_post_discount_sales_lcb95": -2.5}
    assert objective_reward(row, "post_discount_sales") == (-1.0, -2.5)
    assert objective_reward(row, "list_value") == (4.0, 3.0)
    with pytest.raises(ValueError):
        objective_reward(row, "profit")
