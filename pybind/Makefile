MOD_NAME = _diffusion_maps

# Flags

CPPFLAGS_RELEASE = -DNDEBUG

CXXFLAGS_BASE    = -std=c++17 -pedantic -Wall -Wextra -Werror -I../include $(shell python3-config --includes)
CXXFLAGS_DEBUG   = -g -fsanitize=address -fsanitize=undefined
CXXFLAGS_TEST    = $(CXXFLAGS_DEBUG) -Og
CXXFLAGS_RELEASE = -O3 -funroll-loops -march=native

LDLIBS_DEBUG = -lasan -lubsan
LDLIBS_TEST  = $(LDLIBS_DEBUG)

# ------------------------------------------------------------------------------

CPPFLAGS = $(CPPFLAGS_BASE) $(CPPFLAGS_$(PROFILE))
CXXFLAGS = $(CXXFLAGS_BASE) $(CXXFLAGS_$(PROFILE))
LDLIBS   = $(LDLIBS_BASE) $(LDLIBS_$(PROFILE))

MOD = $(MOD_NAME)$(shell python3-config --extension-suffix)
BUILD_DIR = build/$(PROFILE)
LIB = ../build/$(PROFILE)/diffusion_maps.a

.PHONY: all clean-profile clean

all: $(BUILD_DIR)/$(MOD)

$(BUILD_DIR)/$(MOD): $(BUILD_DIR)/mod.o
	$(CXX) $(LDFLAGS) $^ $(LIB) $(LDLIBS) -shared -o $@

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -fPIC -MMD -MF $(BUILD_DIR)/$*.d -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean-profile:
	$(RM) -r $(BUILD_DIR)

clean:
	$(RM) -r build

-include $(BUILD_DIR)/*.d
