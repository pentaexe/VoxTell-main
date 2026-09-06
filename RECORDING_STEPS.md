# Recording steps — literal

Your script is `voxtell-plugin/DEMO.md`. This file is just the mechanics.

---

## A. Before you press record (5 min)

1. **Open DEMO.md on a second screen** — or print it. Do not read it off the
   screen you are recording.

2. **Close everything noisy.** Slack, Outlook, Teams. A notification banner
   mid-take means starting over.

3. **Open VS Code** on `C:\Users\brian\OneDrive\Desktop\Code\VoxTell-main`.

4. **Start a NEW Claude Code conversation.** Not this one — a fresh session, so
   nothing is already in context.

5. **Type `/mcp` and press Enter.** You need to see `voxtell-seg` listed as
   connected. If it is not there, the tools will not work: restart VS Code and
   check again before going further.

6. **Make the text bigger.** `Ctrl` + `+` two or three times. It will look
   oversized to you and correct in the video.

7. **Open a terminal** in VS Code: ``Ctrl` `` (backtick). You need it for steps
   C4 and C5. Leave it open but collapsed.

8. **Do one full rehearsal.** Especially step C3 — it loads a 4-billion
   parameter model and sits for about 30 seconds. Know that pause is coming.

---

## B. Recording

- **Start:** `Win` + `Alt` + `R`. A small bar appears with a timer.
- **Stop:** `Win` + `Alt` + `R` again.
- **Saved to:** `C:\Users\brian\Videos\Captures`
- **Named after** the focused window, so it will say `VoxTell-main - Visual Studio Code`.

Game Bar records **the focused window**, not the whole screen. If you click
another app mid-take it may stop. Stay in VS Code.

Record the whole thing in **one take**. Do not stop between steps.

---

## C. What to type, in order

Talk over each one — the lines are in DEMO.md under the matching Act.

**C1.** Say the Act 1 framing out loud first, then type:

```
Can you segment a brain tumor from CT_imagesVal/FLARETs_0001_0000.nii.gz?
```

Wait for `Request: REFUSED`. Read the reasoning aloud. Make the Act 2 point:
this is a tool result, not advice — no model loaded, no GPU touched.

**C2.**

```
What about the spleen in that same image?
```

`Request: ALLOWED`. Make the Act 3 point: not a blanket block.

**C3.**

```
Segment the brain from the MNI image in Downloads.
```

**This is the ~30 second pause.** Fill it with the Act 2 explanation of how
validation decided — Hounsfield units, air fraction. Then read the voxel count
when it lands and note it matches the INT4 measurement.

**C4.** Open the terminal (``Ctrl` ``) and show the structure:

```
tree /F voxtell-plugin /A
```

Walk the three mechanisms while it is on screen — Act 5 in DEMO.md.

**C5.** Still in the terminal:

```
& "C:\Users\brian\.vscode\extensions\anthropic.claude-code-2.1.251-win32-x64\resources\native-binary\claude.exe" plugin validate .\voxtell-plugin --strict
```

`√ Validation passed`.

**C6.** Optional, only if you want the ChatGPT half on camera:

```
& "C:\Users\brian\miniconda3\envs\voxtell\python.exe" voxtell-plugin\mcp_server\server.py --http --port 8765
```

New terminal tab:

```
curl -X POST http://127.0.0.1:8765/mcp -H "Content-Type: application/json" -d '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\",\"params\":{}}'
```

Then `Ctrl+C` the server. Say that reaching it from ChatGPT needs a tunnel and
an auth layer, and that you have not put it on a public URL.

**C7.** Close with the limits — the "What to say about the limits" section of
DEMO.md. Then stop recording.

---

## D. After

1. Open `C:\Users\brian\Videos\Captures` and watch it back. Check the text is
   readable and the whole answer is visible for each step.

2. If a step is unreadable, re-record only that portion and cut them together,
   or redo the take. One unbroken take is better if you can manage it.

3. Send with a short note: repo link, the three mechanisms in a sentence, and
   the limits list pasted in so it arrives with the video rather than after a
   question.
