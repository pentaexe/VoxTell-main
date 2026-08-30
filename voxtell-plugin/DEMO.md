# Demo script — Skills, MCP, and Plugins

About four minutes. Everything below has been run; nothing is aspirational.

The through-line: **a skill advises, a tool refuses, a plugin ships both.** The
brain-tumour example is what makes that concrete.

---

## Before you start

Open Claude Code in `VoxTell-main`. The project `.mcp.json` loads the server, so
the tools are live — check with `/mcp`, which should list `voxtell-seg`
connected.

Have a terminal ready for the plugin half.

---

## Act 1 — the model does not refuse (30 sec)

Say this, don't show it:

> VoxTell is text-prompted, so it accepts any string. Hand it an abdominal CT
> and ask for a brain tumour and it runs, and returns a mask. The mask is of
> nothing. Nothing in the pipeline says so.

That is the problem. Now the fix.

---

## Act 2 — the tool refuses (45 sec)

Type into Claude Code:

> Can you segment a brain tumor from CT_imagesVal/FLARETs_0001_0000.nii.gz?

It calls `check_request` and comes back:

```
Request: REFUSED
  The prompt asks for a structure in the brain (brain, brain tumor), but the
  image is a CT study of the thorax. That structure is not in this field of
  view, so any mask returned would be meaningless.

Image
  modality : CT
  region   : thorax
  - intensities reach -2048, consistent with Hounsfield units (air ~ -1000)
  - 25.0% of voxels below -700 HU (air)
```

**The point to make:** that is a tool result, not advice. `isError: true`. No
model loaded, no GPU touched. The refusal happened before any compute.

How it decided: intensities reaching −2048 mean Hounsfield units, so CT. A
quarter of the voxels below −700 HU means lungs or bowel gas, so a torso, not a
head. Intensity statistics and geometry — no classifier, nothing to take on
trust.

---

## Act 3 — same image, different prompt (30 sec)

> What about the spleen in that same image?

```
Request: ALLOWED
  Prompt targets the abdomen; the image reads as thorax. These regions often
  share a field of view, so proceeding — check the output covers the target.
```

**Point:** it is not a blanket block. Adjacent regions pass with a caveat, and
anything it cannot classify passes with a note. A validator that refuses
whatever it does not recognise is useless.

---

## Act 4 — it actually segments (45 sec)

> Segment the brain from the MNI image in Downloads.

```
Segmented 'brain'
  device      : cuda:0
  validation  : Prompt targets the brain, and the image looks like a brain study.
  voxels      : 1,812,398
  saved       : ...__brain.npz
```

**Point:** 1,812,398 voxels sits alongside the 1,828,296 measured for the same
prompt in the INT4 comparison. It is a real mask, not plumbing returning
something shaped like one.

---

## Act 5 — the three mechanisms (60 sec)

This is the part Dr. Ma asked for. Show the plugin directory:

```
voxtell-plugin/
├── .claude-plugin/plugin.json   ← Plugin: packaging
├── skills/                      ← Skill: workflow, protocol, standards
│   ├── voxtell-inference/
│   └── nninteractive-inference/
├── .mcp.json                    ← MCP: server registration
└── mcp_server/server.py         ← MCP: four callable tools
```

| | What it is | What it can do |
|---|---|---|
| **Skill** | instructions in context | *tell* the model a brain prompt on abdominal CT is wrong |
| **MCP** | callable tools over JSON-RPC | *make the call fail* |
| **Plugin** | package for both | ship them together, versioned |

The distinction in one line: **the skill knows the rule; only the tool can
enforce it.** One is guidance the model may or may not follow. The other is a
gate it cannot walk past.

Then, in the terminal:

```bash
claude plugin validate ./voxtell-plugin --strict
```
```
√ Validation passed
```

---

## Act 6 — ChatGPT (30 sec)

> MCP has two transports and the protocol is identical on both. stdio needs the
> client to launch the server as a local subprocess — that is Claude Code.
> ChatGPT is hosted and cannot do that, so it needs Streamable HTTP.

```bash
python mcp_server/server.py --http --port 8765
curl -X POST http://127.0.0.1:8765/mcp -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

Same four tools, same refusal, over HTTP. To reach it from ChatGPT you tunnel it
— **and it needs an auth layer first**, because the server reads any local path
it is handed and starts GPU jobs with no authentication.

---

## What to say about the limits

Do not let these be discovered rather than offered:

- **`nninteractive_segment` has never run.** Weights are on the cluster. Same
  session pattern as the benchmarks, so the risk is plumbing, but it is untested.
- **Region detection is thresholds, not a model.** It reads a FLARE abdominal CT
  with lung in frame as `thorax`. Adjacency makes the outcome right; the label is
  imprecise.
- **The HTTP server has no authentication.** Fine on localhost, not fine on a
  public URL.
- **The vocabulary is a word list.** An unrecognised term passes without a check.
  It catches the obvious mistake, not every mistake.

---

## If asked "what was the hardest part"

The bug worth telling: the plugin passed `validate --strict`, loaded without
error, and its tools did not exist. `--plugin-dir` never prompts for config, so
`${user_config.…}` never resolved, the server entry was invalid — and an invalid
entry is skipped **silently**, with nothing in `--debug`. Every check said
healthy while the thing did not work. Found only by calling a tool from a
directory where nothing else could have supplied it.
