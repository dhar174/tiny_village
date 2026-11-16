"""Actual goal-bearing execution through the live simulation update boundary."""
import textwrap
from tests.test_graph_initialization import FreshPythonTestCase
from tests.test_goap_graph_contracts import WORLD_SETUP


class TestActivePlanExecution(FreshPythonTestCase):
    def run_world(self, source):
        return self.run_python(WORLD_SETUP + '\ntry:\n' + textwrap.indent('''
character = next(iter(controller.characters.values()))
from actions import apply_attribute_change
from unittest.mock import patch
strategy = controller.strategy_manager
controller.weather_system['current_weather'] = 'clear'
def set_value(name, value):
    return apply_attribute_change(character, graph, name, lambda _: value)
def full_goal(attribute, value):
    return Goal('Tick goal', character, character, 1, 'tick goal',
        {False: [Condition('tick target', attribute, character, value)]},
        None, None, None, None, None, None, [], graph, 'Basic Needs',
        {attribute: value})
''' + textwrap.dedent(source), '    ') + '\nfinally:\n    controller.close()\n')

    def test_subsecond_update_does_not_execute(self):
        self.run_world('''
before = controller.game_statistics['actions_executed']
controller.update_game_state(.1)
assert controller.game_statistics['actions_executed'] == before, 'action before interval'
assert character.energy == graph.G.nodes[character]['energy'] == 20
''')

    def test_fixed_goal_rest_work_across_intervals(self):
        self.run_world('''
set_value('job', 'Baker')
set_value('wealth_money', 5)
from tiny_strategy_manager import PlanRequest
request = PlanRequest(character.uuid, full_goal('wealth_money', 25), source='test')
controller.queue_plan_request(request)
with patch.object(strategy, 'plan_for_goal', wraps=strategy.plan_for_goal) as planning:
    controller.update_game_state(.5)
    assert planning.call_count == 0
    controller.update_game_state(.5)
    first = controller.get_plan_execution_state(character)
    assert first['cursor'] == 1 and first['phase'] == 'RUNNING', first
    assert character.energy == graph.G.nodes[character]['energy'] == 35
    assert character.wealth_money == 5
    controller.update_game_state(1)
    final = controller.get_plan_execution_state(str(character.uuid))
    assert final['phase'] == 'COMPLETE' and final['cursor'] == 2, final
    assert final['goal'] is not None and planning.call_count == 1
assert character.energy == graph.G.nodes[character]['energy'] == 5
assert character.wealth_money == graph.G.nodes[character]['wealth_money'] == 25
assert request.goal.check_completion()
print({'trace': ['Rest', 'Work'], 'money': character.wealth_money, 'phase': final['phase']})
''')

    def test_daily_request_is_pure_and_default_plan_completes(self):
        self.run_world('''
with patch.object(strategy, 'plan_for_goal', wraps=strategy.plan_for_goal) as planning:
    request = strategy.create_daily_request(character)
    assert planning.call_count == 0
controller.queue_plan_request(request)
set_value('current_satisfaction', 0)
for index in range(8):
    controller.update_game_state(1)
    status = controller.get_plan_execution_state(character)
    assert status['cursor'] == index + 1, status
assert status['phase'] == 'COMPLETE', status
assert character.energy == graph.G.nodes[character]['energy'] == 80
assert character.current_satisfaction == graph.G.nodes[character]['current_satisfaction'] == 80
print({'daily_trace':[action.name for action in status['actions']], 'energy':character.energy, 'satisfaction':character.current_satisfaction, 'phase':status['phase']})
before = controller.game_statistics['actions_executed']
controller.update_game_state(1)
assert controller.game_statistics['actions_executed'] == before
''')

    def test_request_outcomes_and_fixed_goal_are_truthful(self):
        self.run_world('''
from tiny_strategy_manager import PlanRequest
original = full_goal('energy', 30)
request = PlanRequest(character.uuid, original, source='test')
original.completion_conditions[False][0].satisfy_value = 99
proposal = strategy.plan_request(request)
assert proposal.outcome == 'PLAN' and proposal.actions
set_value('energy', 30)
assert strategy.plan_request(request).outcome == 'SATISFIED'
set_value('tokens', 0)
assert strategy.plan_request(PlanRequest(character.uuid, full_goal('tokens', 1))).outcome == 'NO_PLAN'
assert strategy.plan_request(PlanRequest(character.uuid, full_goal('missing_scalar', 1))).outcome == 'ERROR'
''')

    def test_delayed_update_coalesces_and_pause_preserves_fraction(self):
        self.run_world('''
before = controller.game_statistics['actions_executed']
controller.update_game_state(.5)
controller.paused = True
controller.update_game_state(30)
controller.paused = False
controller.update_game_state(2.75)
assert controller.game_statistics['actions_executed'] == before + 1
assert abs(controller.get_plan_execution_state(character)['accumulator'] - .25) < 1e-9
controller.update_game_state(.74)
assert controller.game_statistics['actions_executed'] == before + 1
controller.update_game_state(.01)
assert controller.game_statistics['actions_executed'] == before + 2
for value in (-1, float('nan'), float('inf')):
    try: controller.update_game_state(value)
    except ValueError: pass
    else: raise AssertionError('invalid dt admitted')
''')

    def test_priority_preempts_and_resumes_with_fresh_plan(self):
        self.run_world('''
from tiny_strategy_manager import PlanRequest
set_value('job', 'Baker'); set_value('wealth_money', 5)
ordinary = PlanRequest(character.uuid, full_goal('wealth_money', 25), priority=(1,0), source='ordinary')
urgent = PlanRequest(character.uuid, full_goal('energy', 50), priority=(9,0), source='urgent')
controller.queue_plan_request(ordinary)
controller.queue_plan_request(PlanRequest(character.uuid,full_goal('energy',80),priority=(1,0),source='equal'))
controller.queue_plan_request(PlanRequest(character.uuid,full_goal('energy',95),priority=(0,0),source='lower'))
with patch.object(strategy, 'plan_for_goal', wraps=strategy.plan_for_goal) as planning:
    controller.update_game_state(1)
    assert character.energy == 35
    assert controller.get_plan_execution_state(character)['source'] == 'ordinary'
    controller.queue_plan_request(urgent)
    controller.update_game_state(.5)
    assert character.energy == 35
    controller.update_game_state(.5)
    assert character.energy == 50
    assert controller.get_plan_execution_state(character)['source'] == 'urgent'
    controller.update_game_state(1)
    status = controller.get_plan_execution_state(character)
    assert status['source'] == 'ordinary' and status['phase'] == 'COMPLETE', status
    assert planning.call_count == 3
    assert status['pending'] == ('equal', 'lower')
assert character.wealth_money == 25
''')

    def test_failed_generated_action_gets_real_multistep_replacement(self):
        self.run_world('''
from tiny_strategy_manager import PlanRequest
set_value('job', 'Baker'); set_value('wealth_money', 5)
controller.queue_plan_request(PlanRequest(character.uuid, full_goal('wealth_money',25)))
invoke = controller._invoke_action_execute
attempts = []
def fail_rest(action, actor):
    attempts.append(action.name)
    return False if action.name == 'Rest' else invoke(action,actor)
with patch.object(controller, '_invoke_action_execute', side_effect=fail_rest):
    controller.update_game_state(1)
    assert controller.get_plan_execution_state(character)['cursor'] == 0
    assert len(attempts) == 1
    for _ in range(3): controller.update_game_state(1)
assert attempts == ['Rest', 'Exercise', 'Work as Baker'], attempts
assert controller.get_plan_execution_state(character)['phase'] == 'COMPLETE'
assert character.wealth_money == graph.G.nodes[character]['wealth_money'] == 25
assert controller.game_statistics['actions_failed'] == 1
''')

    def test_no_alternative_blocks_until_cooldown_and_relevant_change(self):
        self.run_world('''
from tiny_strategy_manager import PlanRequest
set_value('tokens', 0)
action = Action('Gain token', [], [{'targets':['initiator'],'attribute':'tokens','change_value':1}], graph_manager=graph)
controller.queue_plan_request(PlanRequest(character.uuid, full_goal('tokens',1), extra_actions=(action,)))
with patch.object(controller, '_invoke_action_execute', return_value=False):
    controller.update_game_state(1)
controller.update_game_state(1)
assert controller.get_plan_execution_state(character)['phase'] == 'BLOCKED'
with patch.object(strategy, 'plan_request', wraps=strategy.plan_request) as planning:
    for _ in range(6): controller.update_game_state(1)
    assert planning.call_count == 0, planning.call_count
    set_value('unrelated_741', 1)
    controller.update_game_state(1)
    assert planning.call_count == 0, 'unrelated scalar woke blocked plan'
    set_value('tokens', .25)
    controller.update_game_state(1)
    assert planning.call_count == 1
    assert character.__dict__['tokens'] == .25
    controller.update_game_state(1)
assert controller.get_plan_execution_state(character)['phase'] == 'COMPLETE'
assert graph.G.nodes[character]['tokens'] == 1.25
''')

    def test_success_without_effects_is_not_credited(self):
        self.run_world('''
from tiny_strategy_manager import PlanRequest
set_value('tokens', 0)
action = Action('Gain token', [], [{'targets':['initiator'],'attribute':'tokens','change_value':1}], graph_manager=graph)
controller.queue_plan_request(PlanRequest(character.uuid,full_goal('tokens',1),extra_actions=(action,)))
before = controller.game_statistics['actions_executed']
with patch.object(controller, '_invoke_action_execute', return_value=True) as invoke:
    controller.update_game_state(1)
    assert invoke.call_count == 1
status = controller.get_plan_execution_state(character)
assert status['cursor'] == 0 and status['phase'] == 'RECOVERING', status
assert controller.game_statistics['actions_executed'] == before
assert not status['goal'].check_completion()
''')

    def test_event_requests_are_pure_and_explicit_actor_is_preserved(self):
        self.run_world('''
events = [{'type':'crisis','importance':9,'impact':-10,'participants':[str(character.uuid)]}]
with patch.object(strategy,'plan_for_goal',wraps=strategy.plan_for_goal) as planning:
    requests = strategy.create_event_requests(events,default_subjects=tuple(controller.characters.values()))
    assert planning.call_count == 0
assert len(requests) == 1 and requests[0].actor_uuid == character.uuid
assert requests[0].extra_actions and requests[0].priority == (9,10)
errors = []
controller._apply_strategy_result(requests, errors)
assert not errors
assert controller.game_statistics['actions_executed'] == 0
try: strategy.create_event_requests([{'type':'work','participants':['missing actor']}],default_subjects=(character,))
except (ValueError,KeyError): pass
else: raise AssertionError('bad explicit actor broadcast')
''')

    def test_load_and_manager_replacement_clear_scheduling(self):
        self.run_world('''
import tempfile, pathlib
request = strategy.create_daily_request(character)
controller.queue_plan_request(request)
controller.update_game_state(.5)
with tempfile.TemporaryDirectory() as directory:
    path = pathlib.Path(directory)/'state.json'
    path.write_text('{"statistics": {}}')
    assert path.read_text() == '{"statistics": {}}'
    assert controller.load_game_state(str(path))
assert controller.get_plan_execution_state(character)['phase'] == 'IDLE'
controller.update_game_state(.5)
assert controller.game_statistics['actions_executed'] == 0
controller.strategy_manager = type(strategy)(graph_manager=graph,use_llm=False)
controller.update_game_state(.5)
assert controller.get_plan_execution_state(character)['accumulator'] == .5
''')

    def test_legacy_replan_returns_bool_and_executes_complete_replacement(self):
        self.run_world('''
set_value('tokens', 0)
first = Action('Earn token', [], [{'targets':['initiator'],'attribute':'tokens','change_value':1}], initiator=character, graph_manager=graph)
plan = Plan('tokens', [full_goal('tokens',2)], graph_manager=None)
plan.add_action(first)
assert plan.replan() is True
assert len(plan.action_queue) == 2
assert plan.execute() is True
assert character.tokens == graph.G.nodes[character]['tokens'] == 2
assert plan.replan() is True and not plan.action_queue
set_value('tokens', 0)
plan = Plan('blocked', [full_goal('tokens',2)], graph_manager=graph)
plan.add_action(first)
alternative = plan.find_alternative_action(first)
assert alternative is None
assert plan.handle_failure(first) is False and not plan.action_queue
plan = Plan('exhausted', [full_goal('tokens',2)], graph_manager=graph)
plan.add_action(first); plan._replan_count = plan._max_replans
assert plan.replan() is False and not plan.action_queue
''')

    def test_partial_write_failure_recovers_from_actual_state(self):
        self.run_world('''
from tiny_strategy_manager import PlanRequest
set_value('tokens',0)
partial = Action('Partial',[],[{'targets':['initiator'],'attribute':'tokens','change_value':1},
    {'targets':['initiator'],'attribute':'temp_741','change_value':1}], cost=.1, graph_manager=graph)
alternative = Action('Alternative',[],[{'targets':['initiator'],'attribute':'tokens','change_value':1}], cost=.2,graph_manager=graph)
controller.queue_plan_request(PlanRequest(character.uuid,full_goal('tokens',2),extra_actions=(partial,alternative)))
original = graph.update_node_attribute
def fail_second(node,attribute,value):
    original(node,attribute,value)
    if attribute == 'temp_741': raise RuntimeError('second write failed')
with patch.object(graph,'update_node_attribute',side_effect=fail_second):
    controller.update_game_state(1)
assert character.tokens == graph.G.nodes[character]['tokens'] == 1, (character.tokens, controller.get_plan_execution_state(character))
assert 'temp_741' not in graph.G.nodes[character]
controller.update_game_state(1)
assert character.tokens == 1
controller.update_game_state(1)
assert character.tokens == 2, (character.tokens, controller.get_plan_execution_state(character))
assert controller.get_plan_execution_state(character)['phase'] == 'COMPLETE'
assert controller.game_statistics['actions_failed'] == 1
''')

    def test_graph_change_invalidates_tail_before_execution(self):
        self.run_world('''
from tiny_strategy_manager import PlanRequest
set_value('job','Baker'); set_value('wealth_money',5)
controller.queue_plan_request(PlanRequest(character.uuid,full_goal('wealth_money',25)))
controller.update_game_state(1)
graph.G.nodes[character]['energy'] = 0
assert character.energy == 35
before = controller.game_statistics['actions_executed']
controller.update_game_state(1)
assert controller.game_statistics['actions_executed'] == before
assert graph.G.nodes[character]['energy'] == 0
controller.update_game_state(1)
assert character.energy == graph.G.nodes[character]['energy'] == 15
for _ in range(3):
    if controller.get_plan_execution_state(character)['phase'] == 'COMPLETE': break
    controller.update_game_state(1)
assert character.wealth_money == graph.G.nodes[character]['wealth_money'] == 25
assert controller.get_plan_execution_state(character)['phase'] == 'COMPLETE'
''')

    def test_completed_explicit_goal_releases_llm_slot(self):
        self.run_world('''
character.use_llm_decisions = True
controller.queue_plan_request(strategy.create_daily_request(character))
choice = Action('Idle choice',[],[],graph_manager=graph)
with patch.object(strategy,'decide_action_with_llm',return_value=[choice]) as select:
    controller.update_game_state(.5)
    assert select.call_count == 0
    controller.update_game_state(.5)
    assert select.call_count == 0
controller._clear_plan_execution()
from tiny_strategy_manager import PlanRequest
controller.queue_plan_request(PlanRequest(character.uuid,full_goal('energy',50),source='explicit'))
with patch.object(strategy,'decide_action_with_llm',return_value=[choice]) as select:
    controller.update_game_state(1)
    assert controller.get_plan_execution_state(character)['phase'] == 'COMPLETE'
    assert select.call_count == 0
    controller.update_game_state(.5)
    assert select.call_count == 0
    controller.update_game_state(.5)
    assert select.call_count == 1
''')

    def test_condition_operand_containers_are_frozen(self):
        self.run_world('''
from tiny_strategy_manager import PlanRequest
original = full_goal('tokens',0)
condition = original.completion_conditions[False][0]
condition.satisfy_value = [1,2]
request = PlanRequest(character.uuid,original)
condition.satisfy_value.append(3)
assert tuple(request.goal.iter_completion_conditions())[0].satisfy_value == [1,2]
''')

    def test_capacity_receipts_preserve_interrupted_active_request(self):
        self.run_world('''
from tiny_strategy_manager import PlanRequest
ordinary = PlanRequest(character.uuid,full_goal('energy',100),source='ordinary')
controller.queue_plan_request(ordinary)
controller.update_game_state(1)
for index in range(16):
    controller.queue_plan_request(PlanRequest(character.uuid,full_goal('energy',90+index),source=f'pending{index}'))
controller.queue_plan_request(PlanRequest(character.uuid,full_goal('energy',50),source='urgent',priority=(9,0)))
assert controller.plan_request_receipts[-1]['source'] == 'pending15'
assert controller.plan_request_receipts[-1]['phase'] == 'REJECTED_CAPACITY'
controller.update_game_state(1)
status = controller.get_plan_execution_state(character)
assert status['source'] == 'urgent'
assert len(status['pending']) == 16 and 'ordinary' in status['pending']
controller.update_game_state(1)
assert controller.get_plan_execution_state(character)['source'] == 'ordinary'
''')

    def test_time_partitions_have_the_same_daily_trace(self):
        import json
        results = []
        for dt, count in ((1 / 60, 480), (.1, 80)):
            completed = self.run_world(f'''
import json
set_value('current_satisfaction',0)
for _ in range({count}): controller.update_game_state({dt!r})
print('TRACE:' + json.dumps({{'energy':character.energy,'satisfaction':character.current_satisfaction,
    'actions':[entry['action_name'] for entry in controller.action_resolver.execution_history],
    'phase':controller.get_plan_execution_state(character)['phase']}}))
''')
            results.append(json.loads(next(line[6:] for line in completed.stdout.splitlines() if line.startswith('TRACE:'))))
        self.assertEqual(results[0], results[1])
        self.assertEqual(len(results[0]['actions']), 8)

    def test_verifier_exception_invokes_once_and_is_observable(self):
        self.run_world('''
action = Action('Real rest',[],[{'targets':['initiator'],'attribute':'energy','change_value':10}],graph_manager=graph)
def broken_verifier(): raise ValueError('controlled verifier failure')
with patch.object(controller,'_invoke_action_execute',wraps=controller._invoke_action_execute) as invoke:
    assert controller._run_selected_action(action,character,verify_result=broken_verifier) is False
    assert invoke.call_count == 1
assert character.energy == graph.G.nodes[character]['energy'] == 30
assert controller._last_action_result['returned_success'] is True
assert 'controlled verifier failure' in controller._last_action_result['verification_error']
assert controller.game_statistics['actions_executed'] == 0
''')

    def test_object_and_alias_divergence_fail_verification(self):
        self.run_world('''
from tiny_strategy_manager import PlanRequest
set_value('tokens',0)
action = Action('Gain token',[],[{'targets':['initiator'],'attribute':'tokens','change_value':1}],graph_manager=graph)
controller.queue_plan_request(PlanRequest(character.uuid,full_goal('tokens',1),extra_actions=(action,)))
invoke = controller._invoke_action_execute
def corrupt_object(action, actor):
    result = invoke(action, actor)
    actor.tokens = 0
    return result
with patch.object(controller,'_invoke_action_execute',side_effect=corrupt_object):
    controller.update_game_state(1)
assert controller.game_statistics['actions_executed'] == 0
assert controller.get_plan_execution_state(character)['cursor'] == 0
controller._clear_plan_execution()
set_value('wealth_money',5)
graph.G.nodes[character]['money'] = 5
action = Action('Earn',[],[{'targets':['initiator'],'attribute':'money','change_value':20}],graph_manager=graph)
controller.queue_plan_request(PlanRequest(character.uuid,full_goal('wealth_money',25),extra_actions=(action,)))
def corrupt_alias(action, actor):
    result = invoke(action, actor)
    graph.G.nodes[actor]['money'] = 0
    return result
with patch.object(controller,'_invoke_action_execute',side_effect=corrupt_alias):
    controller.update_game_state(1)
assert controller.game_statistics['actions_executed'] == 0
assert controller.get_plan_execution_state(character)['cursor'] == 0
''')

    def test_exception_recovery_is_bounded_without_repeated_search(self):
        self.run_world('''
from tiny_strategy_manager import PlanRequest
set_value('tokens',0)
alternatives = tuple(Action(f'Gain {index}',[],[{'targets':['initiator'],'attribute':'tokens','change_value':1}],cost=index*.1,graph_manager=graph) for index in (1,2,3))
controller.queue_plan_request(PlanRequest(character.uuid,full_goal('tokens',1),extra_actions=alternatives))
with patch.object(controller,'_invoke_action_execute',side_effect=RuntimeError('controlled execution exception')) as invoke:
    for _ in range(5): controller.update_game_state(1)
    assert invoke.call_count == 3
assert controller.get_plan_execution_state(character)['phase'] == 'BLOCKED'
with patch.object(strategy,'plan_request',wraps=strategy.plan_request) as planning:
    for _ in range(6): controller.update_game_state(1)
    assert planning.call_count == 0
assert controller.game_statistics['actions_failed'] == 3
assert character.tokens == graph.G.nodes[character]['tokens'] == 0
''')

    def test_actor_identity_aliases_and_independent_budgets(self):
        self.run_world('''
from tiny_strategy_manager import PlanRequest
other = Character('contractactor',30,energy=20,graph_manager=graph,memory_mode='basic',personality_traits=character.personality_traits,gametime_manager=character.gametime_manager)
graph.add_character_node(other)
controller.characters[other.name] = other
other.use_llm_decisions = False
for actor in (character,other):
    goal = Goal('Recover',actor,actor,1,'recover',{False:[Condition('energy','energy',actor,35)]},
        None,None,None,None,None,None,[],graph,'Basic Needs',{'energy':35})
    controller.queue_plan_request(PlanRequest(actor.uuid,goal))
for identifier in (character,character.uuid,str(character.uuid),character.name):
    assert controller.get_plan_execution_state(identifier)['actor_uuid'] == character.uuid
try: controller.get_plan_execution_state('CONTRACTACTOR')
except ValueError: pass
else: raise AssertionError('ambiguous alias admitted')
controller.update_game_state(.5)
assert controller.game_statistics['actions_executed'] == 0
controller.update_game_state(.5)
assert controller.game_statistics['actions_executed'] == 2
assert character.energy == other.energy == 35
assert all(controller.get_plan_execution_state(actor)['phase'] == 'COMPLETE' for actor in (character,other))
graph.world_state.remove_node(character)
with patch.object(controller,'_invoke_action_execute',wraps=controller._invoke_action_execute) as invoke:
    controller.update_game_state(1)
    assert all(call.args[1] is not character for call in invoke.call_args_list)
assert character.energy == 35
assert character.uuid not in controller._plan_records
''')

    def test_partner_goal_change_wakes_blocked_request(self):
        self.run_world('''
from tiny_strategy_manager import PlanRequest
other = Character('Partner741',30,energy=20,graph_manager=graph,memory_mode='basic',personality_traits=character.personality_traits,gametime_manager=character.gametime_manager)
graph.add_character_node(other)
goal = full_goal('energy',35)
goal.completion_conditions[False].append(Condition('partner_ready','energy',other,35))
controller.queue_plan_request(PlanRequest(character.uuid,goal))
controller.update_game_state(1)
assert controller.get_plan_execution_state(character)['phase'] == 'BLOCKED'
for _ in range(5): controller.update_game_state(1)
apply_attribute_change(other,graph,'energy',lambda _:35)
controller.update_game_state(1)
assert character.energy == 20
controller.update_game_state(1)
assert character.energy == 35
assert controller.get_plan_execution_state(character)['phase'] == 'COMPLETE'
''')

    def test_interval_configuration_is_validated_and_used(self):
        self.run_python('''
from tiny_gameplay_controller import GameplayController
for config in ({'action_interval_seconds':0},{'action_interval_seconds':float('nan')},
               {'retry_cooldown_seconds':-1},{'max_pending_requests':1.5},{'max_recovery_attempts':True}):
    try: GameplayController(config={'headless':True,'planning':config})
    except ValueError: pass
    else: raise AssertionError(config)
''')
        self.run_python(WORLD_SETUP.replace("'headless': True", "'headless': True, 'planning': {'action_interval_seconds':2}") + '''
try:
 controller.update_game_state(1)
 assert controller.game_statistics['actions_executed'] == 0
 controller.update_game_state(1)
 assert controller.game_statistics['actions_executed'] == 1
finally: controller.close()
''')

    def test_graph_replacement_never_invokes_old_bound_action(self):
        self.run_world('''
controller.queue_plan_request(strategy.create_daily_request(character))
controller.update_game_state(.5)
GraphManager._reset_singleton()
controller.graph_manager = GraphManager()
assert controller.graph_manager is not graph
with patch.object(controller,'_invoke_action_execute',wraps=controller._invoke_action_execute) as invoke:
    controller.update_game_state(1)
    assert invoke.call_count == 0
assert not controller._plan_records
assert character.energy == 20
''')

    def test_nested_typed_decisions_preserve_real_action_and_order(self):
        self.run_world('''
set_value('tokens',0)
first = Action('First',[],[{'targets':['initiator'],'attribute':'tokens','change_value':1}],graph_manager=graph)
second = Action('Second',[],[{'targets':['initiator'],'attribute':'tokens','change_value':2}],graph_manager=graph)
decision = {'type':'character_action','character_id':str(character.uuid),'action':first}
errors = []
controller._apply_strategy_result({character.name:[decision,second]},errors)
assert not errors
assert character.tokens == 0
controller.update_game_state(1)
assert character.tokens == 1
controller.update_game_state(1)
assert character.tokens == graph.G.nodes[character]['tokens'] == 3
assert [item['action_name'] for item in controller.action_resolver.execution_history] == ['First','Second']
''')

    def test_native_untargeted_event_uses_explicit_registered_defaults(self):
        self.run_world('''
from datetime import datetime
from tiny_event_handler import Event
event = Event('Village day',datetime.now(),'new_day',5,0,action_system=controller.action_system)
assert event.participants == []
with patch.object(strategy,'plan_for_goal',wraps=strategy.plan_for_goal) as planning:
    requests = strategy.create_event_requests([event],default_subjects=(character,))
    assert planning.call_count == 0
assert len(requests) == 1 and requests[0].actor_uuid == character.uuid
controller.queue_plan_request(requests[0])
controller.update_game_state(1)
assert controller.game_statistics['actions_executed'] == 1
''')

    def test_request_uuid_string_is_canonical_and_coalesced(self):
        self.run_world('''
from tiny_strategy_manager import PlanRequest
first = PlanRequest(character.uuid,full_goal('energy',35))
second = PlanRequest(str(character.uuid),full_goal('energy',35))
assert controller.queue_plan_request(first)
assert not controller.queue_plan_request(second)
assert second.actor_uuid == character.uuid
controller.update_game_state(1)
assert controller.game_statistics['actions_executed'] == 1
''')

    def test_legacy_repeated_occurrence_dependencies_are_not_lost(self):
        self.run_world('''
set_value('tokens',0)
action = Action('Gain',[],[{'targets':['initiator'],'attribute':'tokens','change_value':1}],cost=.1,graph_manager=graph)
plan = Plan('dependencies',[full_goal('tokens',2)],graph_manager=graph)
plan.add_action(action,dependencies=[])
plan.add_action(action,dependencies=['prepare'])
assert plan.replan() is False
assert not plan.action_queue and character.tokens == 0
''')

    def test_legacy_replan_rechecks_previously_completed_goal(self):
        self.run_world('''
set_value('tokens',0)
action = Action('Gain',[],[{'targets':['initiator'],'attribute':'tokens','change_value':1}],cost=.1,graph_manager=graph)
plan = Plan('fresh goal',[full_goal('tokens',2)],graph_manager=graph)
plan.add_action(action)
assert plan.replan() is True and plan.execute() is True
set_value('tokens',0)
assert plan.replan() is True
assert plan.action_queue, 'historical completion substituted for actual predicate'
assert plan.execute() is True
assert character.tokens == graph.G.nodes[character]['tokens'] == 2
''')

    def test_preempted_legacy_sequence_has_explicit_terminal_receipt(self):
        self.run_world('''
from tiny_strategy_manager import PlanRequest
set_value('tokens',0)
first = Action('First',[],[{'targets':['initiator'],'attribute':'tokens','change_value':1}],graph_manager=graph)
second = Action('Second',[],[{'targets':['initiator'],'attribute':'tokens','change_value':2}],graph_manager=graph)
controller._apply_strategy_to_character(character,[first,second])
controller.update_game_state(1)
assert character.tokens == 1
controller.queue_plan_request(PlanRequest(character.uuid,full_goal('energy',35),priority=(9,0)))
controller.update_game_state(1)
assert any(receipt['phase']=='CANCELLED_PREEMPTED' and receipt['cursor']==1 for receipt in controller.plan_request_receipts)
controller.update_game_state(1)
assert character.tokens == 1
assert 'Second' not in [item['action_name'] for item in controller.action_resolver.execution_history]
''')

    def test_recovery_delay_survives_preemption(self):
        self.run_world('''
from tiny_strategy_manager import PlanRequest
set_value('job','Baker'); set_value('wealth_money',5)
controller.queue_plan_request(PlanRequest(character.uuid,full_goal('wealth_money',25)))
with patch.object(controller,'_invoke_action_execute',return_value=False):
    controller.update_game_state(1)
assert controller.get_plan_execution_state(character)['phase']=='RECOVERING'
controller.queue_plan_request(PlanRequest(character.uuid,full_goal('energy',35),priority=(9,0),source='urgent'))
controller.update_game_state(1)
assert character.energy == 35
controller.update_game_state(1)
assert character.wealth_money == 5, 'recovery replacement executed during installation'
controller.update_game_state(1)
assert character.wealth_money == 25
''')


    def test_frozenset_event_participants_resolve_and_execute(self):
        self.run_world('''
from datetime import datetime
from tiny_event_handler import Event
native = Event('Frozen participants', datetime.now(), 'new_day', 5, 0,
    participants=frozenset({character}), action_system=controller.action_system)
payload = {'type': 'new_day', 'participants': frozenset({str(character.uuid)})}
for event in (native, payload):
    requests = strategy.create_event_requests([event])
    assert len(requests) == 1 and requests[0].actor_uuid == character.uuid
    controller.queue_plan_request(requests[0])
assert controller.game_statistics['actions_executed'] == 0
controller.update_game_state(.5)
assert controller.game_statistics['actions_executed'] == 0
controller.update_game_state(.5)
assert controller.game_statistics['actions_executed'] == 1
assert character.energy == graph.G.nodes[character]['energy'] == 35
assert controller.get_plan_execution_state(character)['source'] == 'event:new_day'
''')


class TestVisualPlanExecution(FreshPythonTestCase):
    def test_dummy_sdl_uses_same_interval_and_single_speed_scaling(self):
        self.run_python(WORLD_SETUP.replace("'headless': True", "'headless': False") + '''
try:
 character = next(iter(controller.characters.values()))
 controller.time_scale_factor = 2.
 controller.update_game_state(.5)
 assert controller.game_statistics['actions_executed'] == 0
 controller.update_game_state(.5)
 assert controller.game_statistics['actions_executed'] == 1
 assert character.energy == graph.G.nodes[character]['energy'] == 35
 assert controller.get_plan_execution_state(character)['cursor'] == 1
 controller.render()
finally: controller.close()
''')
