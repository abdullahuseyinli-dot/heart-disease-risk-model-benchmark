# Changelog

All notable changes to HeartShift will be documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioned
software releases are intended to follow [Semantic Versioning](https://semver.org/).

The `legacy-v1-development-consumed` tag preserves historical, non-confirmatory
evidence; it is not a software release represented by this changelog. No stable
release or archive DOI is claimed here.

## [Unreleased]

### Added

- A benchmark-first README, benchmark card, project status, documentation index,
  manuscript evidence index, and an archived record of the original holdout benchmark.
- Cross-platform offline CI, full-evidence validation, full-history secret scanning,
  dependency-licence reporting, a measured coverage floor, and source-distribution
  boundary checks.
- A portable v2 split manifest and clean-wheel installation smoke test.
- Citation metadata for the software and preparatory Zenodo metadata that does
  not claim an existing deposition, release, or DOI.
- A security-reporting policy that avoids public disclosure of vulnerabilities,
  secrets, credentials, or sensitive records.
- Repository-level and file-scoped attribution, licence, transformation, and
  claim-boundary notices for the eICU Collaborative Research Database Demo
  v2.0.1 and the derived smoke-test database.

### Changed

- Reframed the repository around the current HeartShift evidence tracks while
  preserving superseded protocols, failures, raw inputs, and prediction records.
- Aligned installation guidance and package metadata on Python 3.11 and 3.12.
- Disabled checkpoint auto-download in the TabICL estimator path and enforced
  offline model-hub settings in repository workflows.
- Expanded third-party notices to distinguish the repository's MIT-licensed
  software from CC BY 4.0 datasets and the ODbL 1.0 eICU database materials.
