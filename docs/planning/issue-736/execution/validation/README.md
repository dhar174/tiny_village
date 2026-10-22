# Retained validation

`../validation-raw.tar.gz` contains the complete original validation tree (772 files); member SHA-256 values are in raw-member-hashes.json. Selected structured summaries remain expanded here. Absolute scratch paths in original capture metadata identify the original run; archive members use paths relative to this directory.

To inspect all original stdout/stderr, source, timing and terminal records:

```sh
mkdir -p /tmp/issue-736-validation
tar -xzf docs/planning/issue-736/execution/validation-raw.tar.gz -C /tmp/issue-736-validation
```

Map original `/workspace/scratch/issue-736/execution/<subpath>` to the extracted `<subpath>`. The original baseline source export is not included; baseline-main.run.json records its revision, files and results, and git archive can recreate it. No captured failure was removed or converted into a passing result.
