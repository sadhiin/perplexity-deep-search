# TODO

## Completed
- [x] Add a `/chat/metrics` monitoring endpoint backed by `ConversationManager.get_metrics()` so the backend now exposes aggregated conversation, message, and search-session counters (supporting Phase 3.3 monitoring/analytics).
- [x] Rename the conversation metadata column so SQLAlchemy no longer sees the reserved attribute `metadata`; the ORM now stores data in `metadata_json` while exposing a property `metadata` for downstream code/tests.
- [x] Have the research workflow carry the conversation history context into analysis, reporting, and reasoning so the thinking LLM now sees prior chat turns before final report generation (Phase 2.4).
- [x] Ship a `scripts/init_db.py` helper and README instructions so local developers can initialize the SQLite database via SQLAlchemy in one command (Phase 2.1).

## Pending
- [ ] Integration testing for the enhanced research workflows and memory components (Phase 4.1).
- [ ] Technical and user-facing documentation to cover the new dual-LLM architecture, workflow, and UI (Phase 4.2).
- [ ] Performance monitoring surfaces and export/sharing improvements (Phase 3.2 & 3.3).
