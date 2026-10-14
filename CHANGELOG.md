# Changelog

## Unreleased

### Fixed

- Restore the existing fallback Goal when the character import is unavailable
  or circular, preventing utility and GOAP cold imports from raising an
  undefined-Goal NameError (#718, PR #706). Full graph startup remains tracked
  separately in #719.
