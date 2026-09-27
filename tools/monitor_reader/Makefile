PREFIX ?= $(NEKRS_HOME)

.PHONY: all build install clean

all build clean:

install:
	test -n "$(PREFIX)"
	install -d "$(PREFIX)/bin"
	install -m 755 main.py "$(PREFIX)/bin/monitor_reader"
