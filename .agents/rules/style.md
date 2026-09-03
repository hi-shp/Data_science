# Style & Execution Guidelines

1. **No Emojis**: Absolutely do NOT use any emojis in your code, comments, or your chat responses.
2. **Short Comments**: Keep code comments brief and concise. Avoid overly verbose explanations in code.
3. **Auto-run main.py**: Always execute `python3 main.py` in background at the end of each modification cycle.
4. **Full Autonomous Execution**: The user has granted full permissions. Execute all necessary edits, commands, training, and fixes autonomously without asking for approval or confirmation.
5. **Screenshot Naming**: Whenever saving screenshots upon episode completion, always format the filename as `{timestamp}_{SUCCESS/FAIL}.png` (e.g. `YYYYMMDD_HHMMSS_SUCCESS.png` or `YYYYMMDD_HHMMSS_FAIL.png`).
6. **Evaluation Script Policy**: Do NOT continuously modify `test_success_rate.py` during active iterations. Focus all development, tuning, and testing strictly on `main.py` and its core modules. Only update/sync `test_success_rate.py` at the end of the day when the user explicitly requests the overnight test run.
