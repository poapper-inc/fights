PKGS = fights fights-srv
PYTHON = python3
CARGO = cargo

.PHONY: clean

build: fights fights-srv

fights: fights-srv
	cd $@ && $(PYTHON) -m build

fights-srv:
	cd $@ && $(CARGO) build

clean:
	cd fights && $(CARGO) clean
