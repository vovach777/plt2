# Build queue (Claude ↔ Grok builder) — do not merge

This branch and its draft PR are the channel between Claude (writes build tasks) and the Grok builder bot
(builds on Vladimir's computer). Agreed with the bot on 2026-11-13.

- **Task:** a PR comment that starts with `/build <task-name>` (task-name: `YYYY-MM-DD_topic_vN`). A long task is a file
  `queue/tasks/<task-name>.md` in this branch, and the comment is `/build <task-name>` with a link to it.
- **Answer:** a PR comment from the bot that starts with `[grok]`: outcome (success / failed), key numbers, links to
  artifacts. Artifacts stay in Google Drive `builder-bot-grok/results/<task-name>/`.
- **Bot rules:** run the tools exactly as the task says, no additions. If a step does not work as written, use the
  nearest working variant and say so in one line. Never push, merge or message anyone else. Comments go only in this PR.
- Both sides use the `vovach777` account. The `/build` and `[grok]` prefixes tell who wrote what.
- The Drive folder `builder-bot-grok/tasks/` stays as the fallback (the bot polls it).
