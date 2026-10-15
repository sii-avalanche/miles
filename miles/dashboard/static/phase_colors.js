// idle states share a light neutral family (train_wait also gets a hatch
// texture below — the accessibility channel — so idle doesn't lean on hue
// alone); the log_probs family shares one hue at three lightness steps;
// rollout gets the brand orange as the headline phase. Validated against
// the design system's 8-hue categorical order (scripts/validate_palette.js).
export const PHASE_COLORS = {
  initialize: "#ded6ca",
  rollout: "#d55816",
  eval_rollout: "#e8a683",
  actor_train: "#2a78d6",
  train_log_probs: "#c3b8ec",
  log_probs: "#8b7bd8",
  ref_log_probs: "#4a3aa7",
  data_preprocess: "#eda100",
  train_wait: "#efe8dc",
  update_weights: "#e87ba4",
  ref_model_update: "#008300",
  save_model: "#1baf7a",
  sleep: "#e5ddd0",
  wake_up: "#e34948",
};
