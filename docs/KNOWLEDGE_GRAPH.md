# Second brain graph

Atom Zero can turn the local index into a small, explainable knowledge graph.
It is a navigation aid, not a trained NER or reasoning model.

## What it does

- Extracts **capitalized phrases**, **code-like identifiers** (`L-7781`,
  `VB_42`) and **years** from indexed chunks.
- Links two entities when they appear in the **same chunk**.
- Keeps source counts on every node and edge.
- Lets the UI or API reopen the exact evidence chunks for an entity.

The output has:

- `nodes`: display name, chunk mentions and source count;
- `edges`: source entity, target entity, shared-chunk weight and source count;
- `stats`: chunks scanned, nodes, edges, limits;
- `method`: a short description of the statistical extraction.

## CLI

```powershell
atomzero-docs graph --db data/docs.db
atomzero-docs graph --limit-chunks 1000 --min-edge 2 --max-entities 120 `
  --db data/docs.db
atomzero-docs entity Ankara --db data/docs.db
atomzero-docs entity L-7781 --limit 5 --db data/docs.db
```

`graph` prints JSON nodes/edges. `entity` returns the matching chunks,
snippets, page/section locators and the entity's graph connections.

## Chat

The normal assistant also recognizes relation queries:

```text
what is related to Ankara?
what relates to L-7781?
connections for Turkey
graph Ankara
```

The reply contains local snippets, source locators and the entity's strongest
connections, plus a reminder that the evidence is co-occurrence rather than a
semantic relation claim.

## HTTP

```text
GET /v1/graph?limit_chunks=500&min_edge=1&max_entities=80
GET /v1/graph/entity?name=Ankara&limit=10
```

Both endpoints are read-only and local.

## UI

The **Second brain graph** panel shows:

- top entity chips with chunk counts;
- the strongest co-occurrence edges;
- an entity input and **Show evidence** action;
- source/snippet cards with the existing Evidence and Copy citation actions.

## Honest limits

- This is **statistical co-occurrence**, not semantic relation extraction. An
  edge means “appeared together in a chunk”, not “is the capital of” or “works
  for”.
- Capitalization is a heuristic. Lowercase entities, pronouns and sentence
  fragments can be missed or merged.
- Years are treated as entities; this is useful for navigation but not a date
  parser.
- Teacher notes can appear in the graph. Check the source card's `verified`
  state before relying on them.
- The graph is capped by `limit_chunks` and `max_entities`; it is a local
  overview, not an exhaustive database.
- No model weights are involved and no data leaves the machine.

## Why it matters

A document list answers “what did I upload?”. A graph starts to answer “what is
connected to what, and where is the evidence?”. That is a step toward a local
second brain rather than a file browser.
