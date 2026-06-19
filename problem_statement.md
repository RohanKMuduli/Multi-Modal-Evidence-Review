# Problem Statement

Build a claim verification system that evaluates damage claims using:

- a customer claim conversation,
- submitted image paths,
- evidence requirements,
- and user claim history.

For each row in `dataset/claims.csv`, the system must produce a structured output row containing the extracted issue, object part, evidence status, risk flags, claim decision, supporting images, image validity, and severity.

The system should support local deterministic/mock execution for development and optional multimodal provider execution through Gemini or OpenAI.

## Inputs

- `dataset/claims.csv`: test claims to process.
- `dataset/sample_claims.csv`: development examples with expected outputs.
- `dataset/user_history.csv`: historical user context and risk flags.
- `dataset/evidence_requirements.csv`: minimum evidence standards.
- `dataset/images/sample/`: images referenced by `sample_claims.csv`.
- `dataset/images/test/`: images referenced by `claims.csv`.

## Required Output

The pipeline writes `output.csv` with the following columns:

```text
user_id,image_paths,user_claim,claim_object,evidence_standard_met,
evidence_standard_met_reason,risk_flags,issue_type,object_part,
claim_status,claim_status_justification,supporting_image_ids,
valid_image,severity
```

## Decision Labels

- `supported`: submitted evidence supports the claim.
- `contradicted`: submitted evidence clearly conflicts with the claim.
- `not_enough_information`: evidence is insufficient or unclear.
