# Astro Dependency Policy

Astro modules are arranged into clear tiers so that imports remain acyclic and foundational utilities stay free of higher-level concerns.

## Tiers

1. **Foundations**
   - Modules: `astro.typings.base`, `astro.typings.outputs`, everything under `astro.utilities`, and other helper modules that only depend on the standard library or third-party packages.
   - Rules: no logging, no imports from infrastructure or domain layers. Circular dependencies must be resolved here, not patched at higher layers.

2. **Infrastructure**
   - Modules: logging (`astro.logger`), configuration (`astro.paths`, `astro.config`), and similar support code.
   - Rules: may import foundations, but never domain/agents. These modules provide services (logging, config) to upper layers.

3. **Domain / Interface**
   - Modules: agents, CLI, application logic, tests, and integration layers.
   - Rules: may import infrastructure and foundations. When adding new functionality, prefer exposing shared helpers in infrastructure or foundations rather than creating horizontal dependencies.

## Allowed Import Directions

- Foundations ➜ Foundations only.
- Infrastructure ➜ Foundations.
- Domain ➜ Infrastructure and Foundations.
- Reverse edges are prohibited (e.g., Infrastructure must never import Domain).

## Design Checklist

When introducing a new module:

1. Decide which tier it belongs to and ensure all imports comply with the allowed direction.
2. If it needs helpers from a lower tier, add them to the appropriate foundation module (for example `astro.typings.base` for functional helpers, `astro.utilities` for pure utilities).
3. If logging or configuration is required, implement a wrapper in the infrastructure tier; keep low-level utilities pure.
4. Run the linters/tests and a dependency check (see below) before committing.

## Tooling

Use [`import-linter`](https://github.com/seddonym/import-linter) (or a similar tool) to enforce these contracts. Suggested contract definitions:

```ini
[contract:foundations_no_infrastructure]
name = Foundations must not depend on Infrastructure
type = forbidden
modules =
    astro.typings
    astro.utilities
forbidden_modules =
    astro.logger
    astro.agents
    astro.app

[contract:infrastructure_no_domain]
name = Infrastructure must not depend on Domain
type = forbidden
modules =
    astro.logger
    astro.paths
    astro.config
forbidden_modules =
    astro.agents
    astro.app
```

Add the lint command to CI once the contracts are passing to keep the dependency graph healthy.
