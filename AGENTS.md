# Project Instructions

## Project goal

This repository is an AI hackathon project.

Task of the Project
"The service takes order parameters and returns up to 3 contractor cards, each with an explanation of why this contractor is there. The value lies in the explanation, not in the sorting."

The goal is to build a working prototype quickly while keeping the code
simple and readable.

Prefer a working simple solution over an unnecessarily complex architecture.

## Tech stack

Primary stack:

- Python
- Fast API
- PostgreSQL
- HTML, CSS, JavaScript

## General coding rules

- Prefer simple and readable code over clever code.
- Use descriptive variable and function names.
- Keep functions small and focused on one responsibility.
- Use Python type hints.
- Avoid unnecessary classes and abstractions.
- Avoid duplicated code when a small reusable function is enough.
- Do not over-engineer the project.
- Do not create microservices unless explicitly requested.
- Do not rewrite working code unless necessary.
- Preserve existing behavior unless the task requires changing it.
- Do not remove existing features without permission.

The developers of this project are still learning, so code should be
easy to understand and modify.

## Before changing code

Before implementing a task:

1. Inspect the relevant existing files.
2. Understand how the current implementation works.
3. Identify the smallest reasonable change.
4. Briefly state the implementation plan.
5. Then make the changes.

Do not guess the structure of files that you have not inspected.

## AI integration rules

AI calls should normally be isolated from FastAPI route handlers.

Keep prompts in one clearly identifiable place when practical.

Never expose API keys.

Read secrets from environment variables such as:

OPENAI_API_KEY

If AI output is expected to follow a structure, validate it instead of
blindly trusting model output.

Handle API failures and invalid responses gracefully.

## Database rules

## Dependencies

Do not add new dependencies unless they provide clear value.

Before installing a new package:

1. Check whether the existing standard library or installed dependencies
   can solve the problem.
2. If a new dependency is needed, explain why.
3. Add it to requirements.txt.

Do not change the Python version unless necessary.

## Security

Never:

- hardcode API keys
- commit .env files
- expose passwords or tokens
- log secrets
- place credentials in frontend JavaScript

Use environment variables for secrets.

Ensure `.env` is included in `.gitignore`.

## Scope control

When given a task, modify only the files necessary for that task.

Do not:

- redesign the whole application
- rename unrelated files
- perform large refactors
- replace technologies
- change working APIs

unless explicitly requested.

## When requirements are unclear

If an ambiguity could significantly change the product behavior or
architecture, ask before making the decision.

For small implementation details, choose the simplest reasonable option.

## Final response

After completing a task, briefly report:

1. What was changed.
2. Which files were changed.
3. How to run or test it.
4. Any important limitations or remaining problems.

Keep explanations understandable for junior Python developers.