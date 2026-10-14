# Issue #718: Goal fallback validation

Reviewed on 2026-10-13 for [#718](https://github.com/dhar174/tiny_village/issues/718)
and [PR #706](https://github.com/dhar174/tiny_village/pull/706).

## Change and revision

The first of two identical `except ImportError` handlers swallowed the failure
to import `tiny_characters.Goal`, leaving `Goal` undefined when utility method
annotations were evaluated. Removing that handler lets the existing fallback
class execute. Its constructor, attributes, and warning remain intact; the real
class is still preferred when it is available during the import.

- Baseline main: `de134632bd4832087dac49f5819a75d2e0aa0747`.
- Original PR head: `c92eb8c71e05b0ffa006bc0067ceb3169e34c8a2`.
- Runtime revision tested after rebasing onto main:
  `cdd6aa3ce2ac57a301d0d487a24ce80f0c5495b1`.
- Subsequent changes in this PR document the results; no further runtime changes.

Both a cold utility import and a cold GOAP import exited 1 on main with
`NameError: name 'Goal' is not defined`. The rebased repair removes that failure.

## Environment and bounds

Commands ran from the repository root with Python 3.12.14 in the existing CPU
environment (`/workspace/.venvs/tiny_village/bin/python`). Each check used a new
subprocess, a 20-second fatal traceback timer, and a 25-second parent timeout.
No heavy model installation or assertion suppression was used.

```bash
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export XDG_CACHE_HOME=/workspace/.cache MPLCONFIGDIR=/workspace/.cache/matplotlib
PY=/workspace/.venvs/tiny_village/bin/python
check() {
    timeout --kill-after=5s 25s "$PY" -c "import faulthandler; faulthandler.dump_traceback_later(20, exit=True); $1"
}
```

The commands below reproduce the executed child commands; the shell `timeout`
provides the same parent deadline used by the validation orchestrator.

```bash
check "import tiny_utility_functions as utility; print('Goal:', utility.Goal.__module__); assert utility.Goal().priority == 0.5"
check "import tiny_goap_system; print('Cold GOAP import passed')"
check "import actions; import tiny_goap_system; print('Actions-first GOAP import passed')"
check "import runpy,sys; sys.argv=['main.py','--mode','minimal','--no-llm','--headless']; runpy.run_path('main.py',run_name='__main__')"
check "import runpy,sys; sys.argv=['main.py','--mode','test','--no-llm','--headless']; runpy.run_path('main.py',run_name='__main__')"
check "import sys,unittest; sys.argv=['unittest','discover','-s','tests','-p','test_goap_error_handling.py','-v']; unittest.main(module=None)"
check "import sys,unittest; sys.argv=['unittest','discover','-s','tests','-p','test_tiny_utility_functions.py','-v']; unittest.main(module=None)"
check "from tiny_globals import get_global_graph_manager; graph=get_global_graph_manager(); assert graph is get_global_graph_manager(); print('Graph singleton initialized')"
```

## Results

| Check | Outcome | Evidence |
| --- | --- | --- |
| Cold utility import | Passed | Actual fallback Goal available; exit 0 |
| Cold GOAP import, including actions-first order | Both passed | Exit 0 for each |
| Character module deliberately unavailable | Passed | Production fallback constructed with name, effects, priority, score, urgency, attributes and defaults |
| Real Goal preference after character module loads | Passed | Reloaded utility binds the actual `tiny_characters.Goal` |
| `main.py --mode minimal --no-llm --headless` | Passed | Exit 0; reduced demo reports **0 actions executed** |
| GOAP error tests supplied by #706 | Passed | 5 executed, 5 passed, 0 skipped |
| `main.py --mode test --no-llm --headless` | Failed | 11 executed: 9 passed, 1 failure, 1 error, 0 skipped |
| Existing utility suite | Failed | 26 executed: 16 passed, 1 failure, 9 errors, 0 skipped |
| Character-first Goal identity diagnostic | Failed assertion | Imports complete, but utility retains the fallback selected during the circular import |
| Global graph initialization | Timed out | Fatal traceback at 20 seconds; recursive acquisition at `tiny_globals.py:226` |

Across the three test suites: **42 executed, 30 passed, 2 failures, 10 errors,
0 skipped**. Import and demo probes are additional functional checks, not unit
test counts. Visual rendering, heavy LLM/model execution, and the full repository
suite were **not executed**. Old CodeQL results on the original PR head do not
validate this revision or establish runtime correctness.

### Fallback probe

Run this in a fresh process under the same timeout. Only the optional character
import is deliberately denied; the utility module and fallback class are real.

```python
import importlib.abc
import sys

class UnavailableCharacters(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tiny_characters":
            raise ImportError("Deliberately unavailable optional character module")

sys.meta_path.insert(0, UnavailableCharacters())
import tiny_utility_functions as utility
goal = utility.Goal("Eat", {"hunger": -0.5}, priority=0.8,
                    urgency=1.0, attributes={"source": "probe"})
assert goal.name == "Eat" and goal.target_effects == {"hunger": -0.5}
assert goal.priority == goal.score == 0.8 and goal.urgency == 1.0
assert goal.attributes == {"source": "probe"}
assert utility.Goal().name == "UnnamedGoal"
assert utility.Goal(score=0.0).score == 0.0
assert callable(utility.calculate_action_utility)
```

Real-class preference was checked with:

```bash
check "import importlib; import tiny_utility_functions as utility; import tiny_characters; importlib.reload(utility); assert utility.Goal is tiny_characters.Goal; print('Real Goal selected when available at import')"
```

The character-first identity probe (`import tiny_characters; import
tiny_utility_functions; assert tiny_utility_functions.Goal is
tiny_characters.Goal`) fails because utility was already imported during the
character import. This repair does not replace an already-selected fallback or
reconcile the different real/fallback Goal constructor interfaces.

## Remaining work and human review

- [#719](https://github.com/dhar174/tiny_village/issues/719): graph startup still
  holds the singleton lock while importing graph -> utility -> characters ->
  Event -> Action -> singleton getter. The second acquisition blocks. No lock
  replacement or graph-lifecycle repair is included here.
- [#773](https://github.com/dhar174/tiny_village/issues/773): integration fixtures
  provide unconfigured `Mock` values for inventory/state. Strategy selection
  slices `food_items[:2]` and errors; fallback action execution returns False.
  Correct fixtures before deciding whether a runtime defect remains.
- [#774](https://github.com/dhar174/tiny_village/issues/774): utility fixtures call
  the real Goal with the fallback-only `priority` interface (9 errors), and their
  MockAction replaces an explicit zero ID (1 failure).

#773 and #774 are separate remediation issues blocked by #718, with scheduling
links to the existing backlog/contract tasks. The import repair is ready for
maintainer review; it does not establish full startup readiness. PR #706 remains
draft, and [#721](https://github.com/dhar174/tiny_village/issues/721) remains the
planned human startup review. No merge, approval, or phase-gate decision was made.
