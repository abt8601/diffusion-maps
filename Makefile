DEFAULT_PROFILE = RELEASE
MOD_NAME = _diffusion_maps

# Flags

CPPFLAGS_DEFAULT = -DNDEBUG

CXXFLAGS_BASE    = -std=c++17 -pedantic -Wall -Wextra -Werror -Iinclude $(shell python3-config --includes)
CXXFLAGS_DEBUG   = -g -fsanitize=address -fsanitize=undefined
CXXFLAGS_DEFAULT = -O3 -funroll-loops -march=native

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
TESTS := $(addprefix $(BUILD_DIR)/,$(basename $(notdir $(wildcard test/test_*))))

ASAN = $(shell ldd $(MOD) | awk '$$1 ~ /^libasan/ { print $$1 }')

# Environment variables passed to pytest when the address sanitizer is used.
# Disable the leak sanitizer because CPython itself leaks memory.
PYTEST_ASAN_ENV = LD_PRELOAD=$(ASAN) ASAN_OPTIONS=detect_leaks=0

.PHONY: all cpptest pytest test clean-mod clean-profile clean

all: $(MOD)

$(MOD): $(BUILD_DIR)/$(MOD)
	cp $< $@

$(BUILD_DIR)/$(MOD): $(BUILD_DIR)/diffusion_maps_pybind.o $(BUILD_DIR)/dummy.o $(BUILD_DIR)/eig_solver.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -shared -o $@

$(BUILD_DIR)/test_%: $(BUILD_DIR)/test_%.o $(BUILD_DIR)/eig_solver.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) $(TEST_LDLIBS) -o $@

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -fPIC -MMD -MF $(BUILD_DIR)/$*.d -o $@

$(BUILD_DIR)/test_%.o: test/test_%.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -fPIC -MMD -MF $(BUILD_DIR)/$*.d -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

test: cpptest pytest

cpptest: $(TESTS)
	for test in $(TESTS); do echo "Running $$test"; $$test; done

pytest: $(MOD)
	$(if $(findstring -lasan,$(LDLIBS)),$(PYTEST_ASAN_ENV)) python3 -m pytest

clean-mod:
	$(RM) $(MOD)

clean-profile:
	$(RM) -r $(BUILD_DIR) $(MOD)

clean:
	$(RM) -r build $(MOD)

-include $(BUILD_DIR)/*.d
