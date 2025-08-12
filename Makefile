.PHONY: test clean

test:
	@echo "Testing WorldModel with different sizes"
	@python test_worldmodel.py --size=1m
	@python test_worldmodel.py --size=12m
	@python test_worldmodel.py --size=25m

test-debug:
	@python test_worldmodel.py --size=debug

clean:
	@find . -name "*.pyc" -exec rm -f {} \;
	@find . -name "__pycache__" -exec rm -rf {} \;