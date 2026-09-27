PREFIX ?= $(NEKRS_HOME)

.PHONY: all build install clean

all build clean:

install:
	test -n "$(PREFIX)"
	install -d "$(PREFIX)/bin" "$(PREFIX)/lib/nekrs-toolbox/dashboard"
	install -m 755 main.py "$(PREFIX)/lib/nekrs-toolbox/dashboard/main.py"
	install -m 644 logreader.py plots.py solver.py metrics.py logs.py controls.py "$(PREFIX)/lib/nekrs-toolbox/dashboard"
	ln -sf ../lib/nekrs-toolbox/dashboard/main.py "$(PREFIX)/bin/nekrs-dashboard"
