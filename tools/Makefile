TOOLS := bp2vtkhdf bptrans monitor_reader
INSTALL_TARGETS := $(addprefix install-,$(TOOLS))
CLEAN_TARGETS := $(addprefix clean-,$(TOOLS))

.PHONY: all build install clean $(TOOLS) $(INSTALL_TARGETS) $(CLEAN_TARGETS)

all build: $(TOOLS)

$(TOOLS):
	$(MAKE) -C $@ all

install: $(INSTALL_TARGETS)

$(INSTALL_TARGETS): install-%:
	$(MAKE) -C $* install

clean: $(CLEAN_TARGETS)

$(CLEAN_TARGETS): clean-%:
	$(MAKE) -C $* clean
