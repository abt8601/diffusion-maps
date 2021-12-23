DEFAULT_PROFILE = DEBUG
MOD_NAME = _diffusion_maps

# Flags

CPPFLAGS_RELEASE = -DNDEBUG

CXXFLAGS_BASE    = -std=c++17 -pedantic -Wall -Wextra -Werror -Iinclude
CXXFLAGS_DEBUG   = -g -fsanitize=address -fsanitize=undefined
CXXFLAGS_RELEASE = -O3 -funroll-loops -march=native

LDLIBS_DEBUG = -lasan -lubsan

TEST_LDLIBS_BASE = -lcriterion

# ------------------------------------------------------------------------------

PROFILE = $(DEFAULT_PROFILE)

CPPFLAGS = $(CPPFLAGS_BASE) $(CPPFLAGS_$(PROFILE))
CXXFLAGS = $(CXXFLAGS_BASE) $(CXXFLAGS_$(PROFILE))
LDLIBS   = $(LDLIBS_BASE) $(LDLIBS_$(PROFILE))

TEST_LDLIBS = $(TEST_LDLIBS_BASE) $(TEST_LDLIBS_$(PROFILE))

MOD = $(MOD_NAME)$(shell python3-config --extension-suffix)
BUILD_DIR = build/$(PROFILE)
TESTS := $(addprefix $(BUILD_DIR)/,$(basename $(notdir $(wildcard tests/test_*))))

ASAN = $(shell ldd $(MOD) | awk '$$1 ~ /^libasan/ { print $$1 }')

# Environment variables passed to pytest when the address sanitizer is used.
# Disable the leak sanitizer because CPython itself leaks memory.
PYTEST_ASAN_ENV = LD_PRELOAD=$(ASAN) ASAN_OPTIONS=detect_leaks=0

.PHONY: all lib pymod cpptest pytest test clean-mod clean-profile clean

all: lib pymod

lib: $(BUILD_DIR)/diffusion_maps.a

$(BUILD_DIR)/diffusion_maps.a: $(BUILD_DIR)/diffusion_maps.o $(BUILD_DIR)/eig_solver.o
	$(AR) $(ARFLAGS) $@ $^

pymod: $(MOD)

$(MOD): $(BUILD_DIR)/diffusion_maps.a
	$(MAKE) -C pybind PROFILE=$(PROFILE)
	cp pybind/build/$(PROFILE)/$(MOD) $@

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -fPIC -MMD -MF $(BUILD_DIR)/$*.d -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

test: cpptest pytest

cpptest: $(BUILD_DIR)/diffusion_maps.a
	$(MAKE) -C tests test PROFILE=$(PROFILE)

pytest: $(MOD)
	$(if $(findstring -lasan,$(LDLIBS)),$(PYTEST_ASAN_ENV)) python3 -m pytest

clean-mod:
	$(RM) $(MOD)

clean-profile:
	$(RM) -r $(BUILD_DIR) $(MOD)
	$(MAKE) -C pybind clean-profile PROFILE=$(PROFILE)
	$(MAKE) -C tests clean PROFILE=$(PROFILE)

clean:
	$(RM) -r build $(MOD)
	$(MAKE) -C pybind clean PROFILE=$(PROFILE)
	$(MAKE) -C tests clean PROFILE=$(PROFILE)

-include $(BUILD_DIR)/*.d
