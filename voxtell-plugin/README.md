# voxtell-seg — a Claude Code plugin

Makes VoxTell and nnInteractive directly callable as segmentation tools, and
refuses requests that are anatomically impossible before spending any compute.

## Why this exists

VoxTell is text-prompted, so it accepts any string. Hand it an abdominal CT and
ask for a brain tumour and it will run, and return a mask. The mask is of
nothing, but nothing in the pipeline says so — you get a plausible-looking
result and no warning.

That is the gap this plugin closes. `check_request` inspects the image and the
prompt independently and reports whether they are compatible; `voxtell_segment`
runs that check first and returns an error instead of a meaningless mask.

```
Request: REFUSED

  The prompt asks for a structure in the brain (brain, brain tumor), but the
  image is a CT study of the thorax. That structure is not in this field of
  view, so any mask returned would be meaningless.

Image
  modality : CT
  region   : thorax
  shape    : (512, 512, 406)   spacing: (0.82, 0.82, 1.25)
  - intensities reach -2048, consistent with Hounsfield units (air ~ -1000)
  - 25.0% of voxels below -700 HU (air)
```

## What each piece does

This plugin bundles all three extension mechanisms, which is the clearest way to
see how they differ:

| Mechanism | What it is | Here |
|---|---|---|
| **Skill** | Instructions loaded into the model's context. Shapes *how it works* — workflow, protocol, standards. Advisory. | `skills/voxtell-inference`, `skills/nninteractive-inference`: cluster procedure, measurement rules, submission standards |
| **MCP server** | A process exposing *callable tools* over JSON-RPC. Real code with real return values. Deterministic — it can refuse. | `mcp_server/server.py`: four tools, including the validation gate |
| **Plugin** | A package bundling skills, MCP servers, commands and hooks into one installable unit | this directory |

The distinction that matters: a skill can *tell* the model that a brain prompt
on an abdominal CT is wrong. Only a tool can *make the call fail*. One is
guidance the model may or may not follow; the other is a gate it cannot walk
past.

Because MCP is an open protocol, the same server works in any MCP client, not
only Claude Code.

## Tools

| Tool | Purpose |
|---|---|
| `check_request` | Validate an image/prompt pair. No compute. Returns inferred modality, region, and the reasoning. |
| `voxtell_segment` | Text-prompted segmentation. Validates first; refuses on mismatch unless `force=true`. |
| `nninteractive_segment` | Bounding-box prompted segmentation. No text, so no anatomy check needed. |
| `list_models` | What is available and what each model can and cannot do. |

## How validation works

Deliberately simple and inspectable — intensity statistics and geometry, not a
classifier. A validator nobody can audit is a validator nobody should trust.

- **Modality** from the intensity distribution. CT is calibrated in Hounsfield
  units, so air sits near -1000; MR intensities are arbitrary and rarely
  negative.
- **Region** from air fraction and field of view. A head has almost no internal
  air; a torso has lungs or bowel gas. Thresholds are stated in `validate.py`.
- **Prompt** against an anatomy vocabulary mapping terms to body regions.

It refuses only on a clear mismatch. Unknown on either side proceeds with a
note — a validator that blocks whatever it cannot classify is useless. Adjacent
regions (thorax/abdomen, abdomen/pelvis) pass, since they routinely share a
field of view.

**Known limits.** The region heuristic is thresholds on one CT convention. It
reads a FLARE abdominal CT with lung in frame as `thorax`, which the adjacency
rule then lets through — correct outcome, imprecise label. It has not been
tested on MR beyond the modality check, and the vocabulary covers common
structures only.

## Install

In Claude Code:

```
/plugin marketplace add pentaexe/VoxTell-main
/plugin install voxtell-seg@voxtell
```

Then, in any conversation:

> Run the setup tool with download true

That checks your machine and fetches the ~1.7 GB checkpoint from Hugging Face
(`mrokuss/VoxTell`) into `~/.cache/voxtell_models`. The checkpoint is not in the
repository, so a fresh install always needs this once.

### Prerequisites

```bash
pip install voxtell nibabel accelerate huggingface_hub
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

A CUDA GPU is effectively required. CPU inference runs but is impractically slow
on a 3-D volume.

`accelerate` looks optional and is not: without it, VoxTell's INT4 loader catches
the ImportError and serves FP16 while still logging INT4.

You can run the same check outside Claude Code:

```bash
python skills/voxtell-inference/scripts/setup_local.py --download
```

### Configuration

All three keys are optional and have working defaults.

| Key | Default | When to change it |
|---|---|---|
| `voxtell_python` | `python` | VoxTell lives in a specific env, e.g. `~/envs/voxtell/bin/python` |
| `voxtell_model_dir` | blank | You already have the checkpoint somewhere; point at the folder holding `plans.json` |
| `nninteractive_weights` | blank | You have the nnInteractive checkpoint (must include `fold_all`) |

### Developing on it

```bash
claude plugin validate ./voxtell-plugin --strict
claude --plugin-dir ./voxtell-plugin
```

> **Every `userConfig` key the MCP server references needs a `default`.**
> `--plugin-dir` does not prompt for configuration, so an unresolved
> `${user_config.…}` makes the server entry invalid, and an invalid entry is
> **skipped silently** with nothing in `--debug` to explain it. The symptom is a
> plugin that validates and loads while its tools simply do not exist.

## Two transports: Claude Code and ChatGPT

MCP's protocol semantics are identical on every transport — a transport only
decides how messages are framed and delivered. So the same four tools are
reachable two ways, and the routing code is written once.

**stdio** (default) — newline-delimited JSON-RPC over the standard streams of a
subprocess the client launches. This is what Claude Code and Claude Desktop use,
and it is what `.mcp.json` configures.

```bash
printf '%s\n' \
 '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
 '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
 | python mcp_server/server.py
```

**Streamable HTTP** — each message is an HTTP POST to one endpoint. A hosted
client such as ChatGPT cannot launch a subprocess on your machine, so stdio is
unavailable to it and this is the binding it needs.

```bash
python mcp_server/server.py --http --port 8765

curl -X POST http://127.0.0.1:8765/mcp -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

`GET /mcp` returns a small status document, which is handy for confirming the
server is up.

> **Before exposing this.** It binds to `127.0.0.1` deliberately. Reaching it
> from a hosted client needs a tunnel (cloudflared, ngrok) *and* an auth layer in
> front. The server reads arbitrary local file paths and starts GPU work, and it
> has no authentication of its own. Do not put it on a public URL as-is.

## Status

Both branches verified end to end through the MCP layer on an RTX 4070 SUPER.

**Allowed request** — MNI T1 with prompt `"brain"`:

```
Segmented 'brain'
  device      : cuda:0
  image       : mni_icbm152_t1_tal_nlin_sym_09a.nii.gz  (197, 233, 189)
  validation  : Prompt targets the brain, and the image looks like a brain study.
  voxels      : 1,812,398
  saved       : ...\mni_icbm152_t1_tal_nlin_sym_09a.nii__brain.npz
```

That voxel count sits alongside the 1,828,296 measured for the same prompt in
the INT4 comparison, so the mask is real rather than an artifact of the plumbing.

**Refused request** — FLARE abdominal CT with prompt `"brain tumor"`: returns
`isError: true`, and no model is loaded and no GPU touched. The refusal happens
before any compute.

`nninteractive_segment` has not been exercised — it needs the weights directory,
which lives on the cluster.
