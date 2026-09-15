# Structured workflows

Workflows are named, bounded research recipes invoked by `reason --workflow`. They use the same retrieval, capability, evidence, and session contracts as the default loop.

| Workflow | Input | Output |
| --- | --- | --- |
| `review` | paper ID or question | structured literature review |
| `gap-analysis` | workspace or question | evidence-linked gaps |
| `impact` | question or session | impact analysis |
| `lineage` | concept | evolution and lineage analysis |

Workflow state is kept in the run context and checkpoints. Tool calls remain subject to the role-scoped tool space and budget policy. Use the default loop when no named recipe matches.
