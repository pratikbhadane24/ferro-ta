# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| latest  | ✅        |

## Reporting a Vulnerability

If you discover a security vulnerability in ferro-ta, please **do not** open a
public GitHub issue.

Instead, report it privately by emailing **pratikbhadane24@gmail.com** with:

- A description of the vulnerability
- Steps to reproduce (or a minimal proof-of-concept)
- The potential impact

You will receive a response within 7 days acknowledging receipt, and a
follow-up within 14 days with next steps.

We will coordinate a fix and public disclosure together. We appreciate
responsible disclosure and will credit researchers who report issues in good
faith.

## Scope

ferro-ta is a numerical computation library. Security-relevant concerns include:

- Memory safety issues in the Rust extension (buffer overflows, use-after-free,
  etc.)
- Unsafe behaviour triggered by crafted input arrays
- Dependency vulnerabilities (tracked via `cargo audit` and Dependabot)

Out of scope: issues in user code that calls ferro-ta, or theoretical attacks
that require direct file-system or network access.

## Hardening and audits

- **Fuzzing:** The project runs `cargo fuzz` targets (e.g. `fuzz_sma`, `fuzz_rsi`) in CI. Crashes are treated as failures; artifacts are uploaded for investigation.
- **Dependency audits:** CI runs `cargo audit` (Rust) and `pip-audit` (Python). Critical and high-severity vulnerabilities should be addressed before release.
- **Reporting:** If you have performed a security assessment or audit, we welcome a private summary to the contact above.
