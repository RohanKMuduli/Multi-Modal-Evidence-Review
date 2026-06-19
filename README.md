# Damage Claim Verification

Multi-modal damage claim verification pipeline for insurance-style claims. The system reads claim text, submitted image paths, historical user risk data, and evidence requirements, then outputs a structured claim decision.

## Repository Layout

```text
.
|-- AGENTS.md
|-- problem_statement.md
|-- README.md
|-- code/
|   |-- main.py
|   |-- src/
|   `-- evaluation/
|       `-- main.py
`-- dataset/
    |-- sample_claims.csv
    |-- claims.csv
    |-- user_history.csv
    |-- evidence_requirements.csv
    `-- images/
        |-- sample/
        `-- test/
```

## Setup

```bash
pip install -r requirements.txt
```

For Gemini runs, set one of these environment variables or create a local `.env` file:

```bash
GEMINI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

For OpenAI runs:

```bash
OPENAI_API_KEY=your_key_here
```

Do not commit `.env` files.

## Run

Run the offline deterministic/mock pipeline on the test input:

```bash
python code/main.py --claims-file dataset/claims.csv --output-file output.csv --mock-mode
```

Run with Gemini:

```bash
python code/main.py --vision-provider gemini --claims-file dataset/claims.csv --output-file output.csv
```

Run sample evaluation:

```bash
python code/evaluation/main.py
```

## Output

The main pipeline writes a CSV with:

- `user_id`
- `image_paths`
- `user_claim`
- `claim_object`
- `evidence_standard_met`
- `evidence_standard_met_reason`
- `risk_flags`
- `issue_type`
- `object_part`
- `claim_status`
- `claim_status_justification`
- `supporting_image_ids`
- `valid_image`
- `severity`

Valid claim statuses are `supported`, `contradicted`, and `not_enough_information`.

## Notes

- Images are treated as the primary source of truth.
- User history only adds risk context; it does not override visual evidence.
- Generated files such as `output.csv`, logs, caches, and evaluation artifacts are ignored by git.
