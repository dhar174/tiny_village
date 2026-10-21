"""Read-only runtime probes for #734; no production substitutes or model loads."""
import contextlib
import json
import os
import shutil
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

@contextlib.contextmanager
def controller(*, visual=False, count=0, performance=None):
    from tiny_gameplay_controller import GameplayController
    from tiny_graph_manager import GraphManager
    original = Path.cwd()
    with tempfile.TemporaryDirectory(prefix='issue734-') as directory:
        os.chdir(directory)
        shutil.copyfile(REPO / "job_roles.json", Path(directory) / "job_roles.json")
        config = {'headless': not visual, 'memory': {'mode': 'basic'},
                  'characters': {'count': count, 'use_llm': False},
                  'checkpoint': {'auto_enabled': False},
                  'map': {'image_path': str(REPO / 'assets/default_map.png')}}
        if performance is not None:
            config['performance_monitoring'] = performance
        instance = None
        try:
            instance = GameplayController(GraphManager(), config)
            yield instance
        finally:
            if instance is not None:
                instance.close()
            os.chdir(original)

def weather():
    import pygame
    from tiny_gameplay_controller import WeatherPanel
    with controller(visual=True) as c:
        c.weather_system = {'current_weather': 'rainy', 'temperature': 15}
        panel = WeatherPanel('audit-weather', position=(10, 10))
        before = pygame.image.tobytes(c.screen, 'RGB')
        height = panel.render(c.screen, c, c.ui_fonts)
        after = pygame.image.tobytes(c.screen, 'RGB')
        assert height > 0 and before != after
        print(json.dumps({'issue': 482, 'weather': 'rainy', 'rendered_height': height,
                          'surface_pixels_changed': True}))

def queue():
    from heapq import heappop
    from tiny_goap_system import Plan, ActionWrapper
    from tiny_graph_manager import GraphManager
    p = Plan('audit-queue', graph_manager=GraphManager())
    a = ActionWrapper('first', cost=1, effects=[], preconditions={})
    b = ActionWrapper('second', cost=1, effects=[], preconditions={})
    p.add_action(a, 1, []); p.add_action(b, 1, ['first'])
    q = list(p.action_queue)
    rows = [heappop(q), heappop(q)]
    assert [r[2].name for r in rows] == ['first', 'second']
    assert all(len(r) == 4 for r in rows)
    assert rows[1][3] == ['first']
    p.replan()
    assert len(p.action_queue) == 2 and all(len(r) == 4 for r in p.action_queue)
    assert next(r for r in p.action_queue if r[2] is b)[3] == ['first']
    print(json.dumps({'issue': 355, 'add_tuple_width': 4, 'initial_ties': 'insertion order',
                      'replanned_tuple_width': 4, 'dependencies_retained': True,
                      'limit': 'Replan ties use id(action); this is not cross-process determinism proof.'}))

def named_target():
    from actions import Action
    with controller(count=2) as c:
        a, b = c.characters.values()
        b.location = a.location
        a.location.current_visitors[:] = [a, b, b]
        old = b.social_wellbeing
        action = Action('audit-encourage', [], [{'targets': ['target_character_in_location'],
            'attribute': 'social_wellbeing', 'change_value': 2}],
            initiator=a, target=b.name, graph_manager=c.graph_manager)
        result = action.execute()
        facts = {'issue': 602, 'result': result, 'before': old,
                 'character_after': b.social_wellbeing,
                 'graph_after': c.graph_manager.G.nodes[b].get('social_wellbeing'),
                 'registered': b in c.graph_manager.G}
        print(json.dumps(facts, default=str), flush=True)
        assert result and b.social_wellbeing == old + 2
        assert c.graph_manager.G.nodes[b].get('social_wellbeing') == b.social_wellbeing

def live_timing():
    import pygame
    with controller(visual=True) as c:
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        c.game_loop()
        summary = c.get_performance_summary()
        print(json.dumps({'issue': 596, 'summary': summary}), flush=True)
        assert summary['frames_observed'] == 1 and summary['last_frame_ms'] >= 0
        assert set(summary['stage_timings_ms']) == {'tick', 'events', 'update', 'render', 'checkpoint'}

def frame_analysis():
    with controller(performance={'history_size': 10}) as c:
        assert c.get_performance_summary()['frames_observed'] == 0
        assert c._analyze_frame_performance(0, {}, 16.67) == []
        normal = c._analyze_frame_performance(5, {'update': 1, 'render': 1}, 16.67)
        assert not any('above the target budget' in s for s in normal)
        for stage in ['render', 'update', 'events', 'checkpoint']:
            suggestions = c._analyze_frame_performance(100, {stage: 80, 'tick': 20}, 16.67)
            assert any('above the target budget' in s for s in suggestions)
            assert len(suggestions) == 2, (stage, suggestions)
        for i in range(15):
            c._record_performance_sample(i + 1, {'render': i + 1}, 60)
        bounded = c.get_performance_summary()
        assert bounded['frames_observed'] == 10
        assert bounded['average_frame_ms'] == 10.5 and bounded['peak_frame_ms'] == 15
        c.performance_metrics['enabled'] = False
        c._record_performance_sample(1000, {'render': 1000}, 60)
        assert c.get_performance_summary() == bounded
        print(json.dumps({'issues': [596, 597], 'bounded_samples': 10,
                          'normal_empty_slow_disabled': 'passed', 'summary': bounded}))

def social():
    with controller(count=2) as c:
        graph = c.graph_manager
        a, b = c.characters.values()
        if graph.G.has_edge(a, b):
            graph.G.remove_edges_from(list(graph.G.edges(a, keys=True)))
        key = graph.G.add_edge(a, b, type='character_character', strength=0.7,
                              emotional=0.4, trust=0.8, interaction_frequency=2.0)
        before = dict(graph.G[a][b][key])
        reported = graph.get_character_relationships(a)
        entry = reported.get(b.uuid, reported.get(b.name))
        c._update_social_relationships(1.0)
        after = dict(graph.G[a][b][key])
        facts = {'issue': 609, 'graph_type': type(graph.G).__name__, 'before': before,
                 'reported': entry, 'after': after}
        print(json.dumps(facts, default=str), flush=True)
        assert entry and entry['strength'] == before['strength'], 'Reader did not expose actual keyed edge attributes'
        assert abs(after['emotional'] - 0.3) < 1e-8 and abs(after['interaction_frequency'] - 1.95) < 1e-8

if __name__ == '__main__':
    globals()[sys.argv[1]]()
