# Skill demo — caption to send with the two clips

Paste this alongside the recordings. Short version first; use the longer one only
if it is going somewhere people will read properly.

---

## Short (for Slack, or an email body)

Two recordings, same question in both: *"I've written a new benchmark script
locally. How do I get it running on Fir?"* Same repo, same model. The only
difference is whether the project skill is loaded.

**With the skill** — it gives the git push-then-pull workflow, the correct venv
(`/home/brianx7/envs/voxtell`) and what breaks without it, and `HF_HOME`, which
compute nodes need because they have no internet.

**Without it** — same repo, so it reads the existing `.sh` files and reconstructs
most of the shape correctly. But it opens with "Fir isn't something I know about"
and then suggests `scp`-ing the file to the cluster, which is the one thing this
project doesn't do: copies drift out of sync, so everything goes through git.

The point isn't that it knows less without the skill. It's that the operational
rules aren't in the code, so it can't follow them. The skill is where they live.

---

## Longer (if it needs to stand on its own)

**Setup.** Both clips are a fresh session, same prompt, same repository contents.
The no-skill clip runs from a copy with the `.claude/` directory removed and no
git history, so nothing can be recovered from previous commits. Everything else
is identical.

**What the skill supplies that the repo does not:**

| | With skill | Without |
|---|---|---|
| Getting code onto the cluster | git push → pull, with the reason | `scp`, which this project doesn't use |
| Virtual environment | correct, plus what fails if wrong | correct, inferred from `.sh` files |
| `HF_HOME` | flagged as mandatory, with why | present but unexplained |
| `seff` | tied to why it matters | listed as a command |

**Why `scp` matters.** It looks reasonable and it is what most projects would do.
Here it is wrong: the cluster checkout has to stay in sync with the repo, and
hand-copied files diverge silently. That rule exists because of a specific
failure, and it is written down in the skill, not in any script.

**What this is worth.** Two of the things the skill states — which venv, and that
`HF_HOME` is required — each cost a failed cluster job to learn. The skill is
where that gets recorded so it costs one job and not three.

**Honest limitation.** Without the skill the model still produced a mostly working
answer, because the repository contains working `.sh` files it can read. The gap
is in the reasoning and the project-specific rules, not in the basic mechanics.
