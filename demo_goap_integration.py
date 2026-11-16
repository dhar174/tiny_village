#!/usr/bin/env python3
"""Execute a real StrategyManager plan and print its authoritative state trace.

This bounded headless example demonstrates #218 sequence execution. It does not
claim #741 autonomous tick scheduling or live LLM inference.
"""

import json
import logging

from actions import Condition, apply_attribute_change
from tiny_characters import Goal
from tiny_gameplay_controller import GameplayController
from tiny_graph_manager import GraphManager
from tiny_strategy_manager import StrategyManager


def demonstrate_goap_integration():
    graph = GraphManager()
    controller = GameplayController(graph, {
        'headless': True, 'memory': {'mode': 'basic'},
        'characters': {'count': 1, 'use_llm': False, 'use_custom': True,
                       'character_data': [{'name': 'PlanningDemo', 'age': 30,
                                           'job': 'unemployed', 'energy': 20}]},
        'checkpoint': {'auto_enabled': False},
    })
    try:
        actor = next(iter(controller.characters.values()))
        apply_attribute_change(actor, graph, 'job', lambda _: 'Baker')
        apply_attribute_change(actor, graph, 'wealth_money', lambda _: 5)
        strategy = StrategyManager(graph_manager=graph)
        goal = Goal('Earn credits', actor, actor, 1, 'earn',
                    {False: [Condition('Earn target', 'wealth_money', actor, 25)]},
                    None, None, None, None, None, None, [], graph,
                    'Economic', {'wealth_money': 25})
        candidates = [action for action in strategy._get_planning_actions(actor)
                      if action.name == 'Rest' or action.name.startswith('Work as')]
        plan = strategy.plan_for_goal(actor, goal, actions=candidates)
        if plan is None:
            raise RuntimeError('No plan found within the configured search bounds')
        trace = []
        for action in plan:
            if not controller._execute_single_action(actor, action):
                raise RuntimeError(f'Action failed: {action.name}')
            state = graph.get_character_state(actor)
            if actor.energy != state['energy'] or actor.wealth_money != state['wealth_money']:
                raise RuntimeError('Character and graph state disagree')
            trace.append({'action': action.name, 'energy': state['energy'],
                          'money': state['wealth_money']})
        achieved = goal.check_completion()
        if not achieved or graph.get_character_state(actor)['wealth_money'] < 25:
            raise RuntimeError('Executed sequence did not reach its original goal')
        result = {'goal': goal.name, 'steps': trace, 'goal_complete': achieved}
        print(json.dumps(result, indent=2))
        return result
    finally:
        controller.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    demonstrate_goap_integration()
