# PR inventory and handoffs

All 37 open PRs are retained at exact snapshot heads. This table assigns investigation handoffs, not keep/rebase/split/supersede or merge approvals. #737 owns current selected-PR CI/ownership/merge ordering; #735 and #736 own their specific reconciliations.

| PR | Title | Snapshot head | Draft | Handoff |
|---|---|---|---|---|
| [#363](https://github.com/dhar174/tiny_village/pull/363) | Add memory summaries to decision prompts | `645ee6472492` | False | #155 |
| [#391](https://github.com/dhar174/tiny_village/pull/391) | Fix happiness calculation test to use flexible threshold instead of hardcoded exact match | `311d65fedc38` | False | #736 |
| [#393](https://github.com/dhar174/tiny_village/pull/393) | Fix GOAPPlanner tests using mock GraphManager instead of None | `e720cc6e45fe` | False | #736 |
| [#395](https://github.com/dhar174/tiny_village/pull/395) | Fix test_goap_implementations to test actual functionality instead of just method existence | `d73a20465eba` | False | #736 |
| [#397](https://github.com/dhar174/tiny_village/pull/397) | Fix test counting logic to properly count individual unittest methods instead of treating TestCase classes as single units | `ec76a141e89d` | False | #736 |
| [#404](https://github.com/dhar174/tiny_village/pull/404) | Fix undefined format_traceback function causing NameError in test failures | `22f5b3b63f1d` | True | #736 |
| [#406](https://github.com/dhar174/tiny_village/pull/406) | Fix hardcoded assertTrue(True) test patterns that always pass | `38b909b07d33` | False | #736 |
| [#408](https://github.com/dhar174/tiny_village/pull/408) | Replace artificial test failures with genuine functional tests in test_fix_validation.py | `72fc125ae0cd` | False | #736 |
| [#417](https://github.com/dhar174/tiny_village/pull/417) | Enhance test_pass method with complex scenarios that can genuinely reveal bugs | `4d0571405de4` | False | #736 |
| [#420](https://github.com/dhar174/tiny_village/pull/420) | Replace artificial module import tests with genuine functionality validation | `264d07e660bb` | False | #736 |
| [#592](https://github.com/dhar174/tiny_village/pull/592) | Add AI Algorithm Mentor workflow | `a2bda6ace451` | False | #770 |
| [#687](https://github.com/dhar174/tiny_village/pull/687) | Align documentation with current demo, testing, and integration behavior | `532bf14edaba` | True | #656 |
| [#688](https://github.com/dhar174/tiny_village/pull/688) | Add current repository context map and refresh folder structure blueprint | `b33d741a9b7b` | True | #633 |
| [#689](https://github.com/dhar174/tiny_village/pull/689) | Align StrategyManager with real GOAP multi-step planning | `93c069420f18` | True | #218 |
| [#690](https://github.com/dhar174/tiny_village/pull/690) | Route EventHandler-driven strategy updates through character-aware execution | `40d344d6cd31` | True | #219 |
| [#691](https://github.com/dhar174/tiny_village/pull/691) | Implement owner-based building access and income effects | `2b0f5a1087d9` | True | #174 |
| [#692](https://github.com/dhar174/tiny_village/pull/692) | Expand default item interactions for tools, clothing, and crafting resources | `e0803c13d5cf` | True | #196 |
| [#693](https://github.com/dhar174/tiny_village/pull/693) | Surface AI-facing inventory state and trade/drop options in prompt context | `6c40b629ef1b` | True | #750 |
| [#694](https://github.com/dhar174/tiny_village/pull/694) | Integrate jobs, item needs, and trading into economic simulation | `d51a7eb5f6f3` | True | #198 |
| [#695](https://github.com/dhar174/tiny_village/pull/695) | Expand item modeling with concrete item types and inventory category support | `b1b889c0bab6` | True | #195 |
| [#696](https://github.com/dhar174/tiny_village/pull/696) | Align action defaults with GraphManager state semantics | `8016a3f46771` | True | #220 |
| [#697](https://github.com/dhar174/tiny_village/pull/697) | Implement job assignment, work outcomes, and career progression in `tiny_jobs.py` | `a3beb371fabd` | True | #199 |
| [#700](https://github.com/dhar174/tiny_village/pull/700) | Remove implemented mini-map TODO from tiny_gameplay_controller.py | `537e6244d432` | True | #770 |
| [#701](https://github.com/dhar174/tiny_village/pull/701) | 🔒 [security fix] Fix world-writable cache directory | `8c19e2bcf6f8` | True | #737 |
| [#702](https://github.com/dhar174/tiny_village/pull/702) | ⚡ Replace hardcoded sleep delays with configurable environment variable | `fb7625547502` | True | #770 |
| [#703](https://github.com/dhar174/tiny_village/pull/703) | 🧪 [testing improvement] Add tests for is_numeric and utilities in tiny_util_funcs.py | `c1528b5bf668` | True | #735 |
| [#704](https://github.com/dhar174/tiny_village/pull/704) | 🧹 Remove unused faiss dependency check in main.py | `56361679188f` | True | #737 |
| [#705](https://github.com/dhar174/tiny_village/pull/705) | Configure supervisor routing decision using LLM | `ba414d9964dd` | True | #770 |
| [#707](https://github.com/dhar174/tiny_village/pull/707) | Implement LLM mocking in test case generation scripts | `7e580e1a1682` | True | #770 |
| [#708](https://github.com/dhar174/tiny_village/pull/708) | Implement functional test boilerplate for invalid input handling in LangGraph generator | `25798624e457` | True | #770 |
| [#709](https://github.com/dhar174/tiny_village/pull/709) | 🧪 Add tests for llm_character_utils.py | `200dfc2d2e96` | True | #737 |
| [#710](https://github.com/dhar174/tiny_village/pull/710) | Implement Map Rendering Fallback | `732010b4172d` | True | #761 |
| [#711](https://github.com/dhar174/tiny_village/pull/711) | 🔒 [security fix] replace silenced exceptions with logging in tiny_prompt_builder.py | `3e2c0c6156f0` | True | #737 |
| [#712](https://github.com/dhar174/tiny_village/pull/712) | ⚡ Optimize A* Pathfinding Membership Checks | `0ece3315a40e` | True | #177 |
| [#713](https://github.com/dhar174/tiny_village/pull/713) | Add anti-aliasing options | `1761d9bd7fd2` | True | #115 |
| [#714](https://github.com/dhar174/tiny_village/pull/714) | 🧪 [testing improvement] Add tests for tweener and is_numeric | `f99d49f9d307` | True | #735 |
| [#715](https://github.com/dhar174/tiny_village/pull/715) | 🧪 Improve testing for find_nodes_within_distance in graph_analytics.py | `6f10f11a0dee` | True | #737 |
