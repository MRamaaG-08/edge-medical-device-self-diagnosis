# Session checklist (print one per session)

Session code: S__    Date: ________   Room temperature: ____ °C   Finger used: ______
Reference HR source: oximeter / manual count (circle)

Before
- [ ] Consent form signed (kept on paper, separately from the data)
- [ ] Hub laptop or Pi on, ESP32 plugged in, `secrets/node_key.txt` present
- [ ] Participant seated, forearm resting on the table, hands warm
- [ ] Clip on the fingertip, firm but not tight; cable taped to the table

Run (two terminals)
- [ ] Terminal 1: `python -m phase2.hardware.hub.live --port COMx --session S__ --protocol`
- [ ] Terminal 2: `python -m phase2.hardware.hub.note_hr --session S__`  (type the reference HR every 30 s)
- [ ] Follow the prompts in terminal 1 (they change every 30 to 60 s; the LED is dimmed automatically)

After
- [ ] Ctrl+C in both terminals; check `data/sessions/S__.jsonl` has about 300 lines
- [ ] `python -m phase2.hardware.hub.store data/sessions/S__.jsonl` (encrypted copy)
- [ ] Copy the four S__ files (.jsonl, .jsonl.enc, _blocks.csv, _refhr.csv) to the team Google Drive folder
- [ ] Notes (anything unusual: sneezed, talked, clip slipped at ~ ___ s):
