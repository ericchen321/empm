This repository hosts a fork of RAI's EMPM codebase.

Environment construction, installation, relinking, qualification, and removal must target only the `.conda/empm` environment; do not select or mutate any other Python environment. 

Follow the "fail-fast" development principle: write only code necessary for achieving
the core functionalities intended by user; no need to overthink about handling of corner
cases, adding gates/checkers/guards against numerical issues unless specifically for
debugging and when added, implement them as warnings/printouts as opposed to exceptions.
Do not add broad exception handlers or silent fallbacks. Avoid unrelated defensive machinery, 
gratuitous hashes, unit tests not critical to assessing core functionalities, and
out-of-scope qualification infrastructure.

Do not create, switch, or delete branches without explicit human instruction. Do
not commit unless separately requested; commits must state that the current human
user and Codex co-authored the code.

Do not modify this document without explicit human approval.
