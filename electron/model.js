// models.js — Central model configuration for all AI providers

const MODELS = {
  gemini: {
    default: "gemini-3.1-flash-lite-preview",
    available: [
      "gemini-3.1-flash-lite-preview",
      "gemini-2.0-flash",
      "gemini-1.5-pro",
    ]
  },
  openai: {
    default: "gpt-4o",
    available: [
      "gpt-4o",
      "gpt-4o-mini",
    ]
  },
  anthropic: {
    default: "claude-3-7-sonnet-20250219",
    available: [
      "claude-3-7-sonnet-20250219",
      "claude-3-5-sonnet-20241022",
      "claude-3-opus-20240229",
    ]
  }
};

// Global default provider
const DEFAULT_PROVIDER = "gemini";

// Global default model (used if no provider or model is specified)
const DEFAULT_MODEL = MODELS[DEFAULT_PROVIDER].default;

/**
 * Get the default model for a given provider.
 * Falls back to gemini-3.1-flash-lite-preview if provider is unknown.
 */
function getDefaultModel(provider) {
  return MODELS[provider]?.default || DEFAULT_MODEL;
}

/**
 * Get all available models for a given provider.
 */
function getAvailableModels(provider) {
  return MODELS[provider]?.available || MODELS[DEFAULT_PROVIDER].available;
}

/**
 * Check if a model is valid for a given provider.
 * If invalid, returns the default model for that provider.
 */
function sanitizeModel(model, provider) {
  const available = getAvailableModels(provider);
  if (!available.includes(model)) {
    console.warn(`Invalid model "${model}" for provider "${provider}". Using default: ${getDefaultModel(provider)}`);
    return getDefaultModel(provider);
  }
  return model;
}

export { MODELS, DEFAULT_PROVIDER, DEFAULT_MODEL, getDefaultModel, getAvailableModels, sanitizeModel };