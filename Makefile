.PHONY: install run validate test clean all

install:
	pip install -r requirements.txt

run:
	python main.py

validate:
	python validate.py

test:
	pytest -q

all: run validate test

clean:
	rm -f preprocessed_tickets.json routing_decisions.json triage_results.json \
	      prediction_comparison.json evaluation_report.json confusion_summary.json \
	      llm_calls.jsonl
	rm -rf llm_outputs/ __pycache__/ pipeline/__pycache__/ tests/__pycache__/
