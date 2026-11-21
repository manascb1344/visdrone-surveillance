# Security

## Reporting a vulnerability

If you find a security issue, please do not open a public issue. Use
GitHub's private vulnerability reporting:

https://github.com/manascb1344/visdrone-surveillance/security/advisories/new

Alternatively, contact the maintainer through their GitHub profile.

## Scope

This project runs machine learning models on video feeds. The main risks are
untrusted input video and model deserialization. Only load model checkpoints
from trusted sources. The `torch.load` calls in this codebase do not use
`weights_only=True`; treat any third-party checkpoint as untrusted code.