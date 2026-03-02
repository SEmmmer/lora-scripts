# WebUI Training Index

This file tracks what training paths are active in WebUI and what has been disabled/cleaned.

## Active (WebUI)

- `sd-lora` -> `./scripts/stable/train_network.py`
- `sdxl-lora` -> `./scripts/stable/sdxl_train_network.py`

Source:

- `mikazuki/app/api.py` (`trainer_mapping`)
- `mikazuki/process.py` (`MODEL_TRAIN_TYPE_TO_TRAINER_FILE`)

## Disabled / Removed from WebUI Path

- `sdxl-finetune` trainer mapping removed from backend entrypoint.
- `mikazuki/schema/dreambooth.ts` removed.
- Schema loading is allowlisted to:
  - `shared.ts`
  - `lora-master.ts`
  - `tagger.ts`
- `mikazuki/schema/lora-basic.ts` removed (route is redirected to `lora/master` in runtime).

## Notes

- Existing `dreambooth` link artifacts in `frontend/dist` are currently handled by runtime hide/redirect logic in `mikazuki/app/application.py`.
- This index is for WebUI routing/entrypoint visibility, not for deleting upstream training scripts from `scripts/stable`.
