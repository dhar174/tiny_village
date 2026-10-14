# Changelog

## Unreleased

### Fixed

- Use validated user-owned Hugging Face caches before optional model imports;
  preserve configured cache locations and remove shared-temp/world-writable
  and machine-specific drive overrides from memory and brain loaders (#720).
- Restore the existing fallback Goal when the character import is unavailable
  or circular, preventing utility and GOAP cold imports from raising an
  undefined-Goal NameError (#718, PR #706).
- Remove import-time sample world construction, synchronize GraphManager
  construction/failure recovery, and retain injected graph ownership through
  domain objects, strategy-generated actions, and controller recovery (#719).
