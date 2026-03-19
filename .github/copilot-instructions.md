- 🧠 Read `.github/instructions/memory-bank.instructions.md` first.
- 🗂 Load all `/memory-bank/*.md` files before each task.
- 📂 Also load files from any active feature subfolder under `/memory-bank/` (for example, `/memory-bank/tasks/`, if present).
- 🚦 Follow the spec-driven workflow described in `.github/instructions/spec-driven-workflow-v1.instructions.md` (PRD → Design → Tasks → Code).
- 🔒 Follow rules in /.github/copilot-rules.md.
- 📝 On "/update memory bank", review **all** Memory Bank files and update them as needed, with extra focus on `activeContext.md`, `progress.md`, and the `tasks/` folder.

When writing tests, do NOT over-mock or fake classes if avoidable. Write tests so that they will fail if the function does not work as expected, do NOT design tests so that they will pass regardless! Good tests fail when there is an error, NEVER manipulate the test design to make it pass while the tested function does not function as expected!

BE CAREFUL AND CONSERVATIVE about creating fake or mock classes as this may not correctly test the functions.

Also, be cautious in test design, only design tests to accurately test functions, do NOT design tests meant to pass even if the function isn't doing exactly what it should do! In other words, don't design tests to pass, design tests that will only pass if the tested code works as intended and fail otherwise.
