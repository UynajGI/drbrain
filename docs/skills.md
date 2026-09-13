# Skills

Skills are instruction packages under `skills/*/SKILL.md`. They describe repeatable CLI workflows and human-facing guidance; they are discoverable as capability descriptors but are intentionally non-executable.

The executable boundary is the CLI or capability adapter named by the skill. Keep skill instructions aligned with the command's `--help` output and update them in the same change as command behavior.

To inspect available skills:

```bash
find skills -mindepth 2 -name SKILL.md -print | sort
uv run drbrain --help
```

Skills do not receive credentials implicitly and cannot bypass capability policy.
