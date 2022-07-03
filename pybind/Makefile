MOD_NAME = _diffusion_maps

CPPFLAGS_TEST_PAR = -DPAR
CPPFLAGS_RELEASE  = -DNDEBUG -DPAR

CXXFLAGS_BASE     = -std=c++17 -pedantic -Wall -Wextra -Werror -I../include $(shell python3-config --includes)
CXXFLAGS_DEBUG    = -g -fsanitize=address -fsanitize=undefined
CXXFLAGS_TEST     = -g -fsanitize=address -fsanitize=undefined -Og
CXXFLAGS_TEST_PAR = -g -fsanitize=thread -fopenmp -Og
CXXFLAGS_RELEASE  = -fopenmp -O3 -funroll-loops -march=native

LDFLAGS_BASE = -L$(LIB_DIR)

LDLIBS_BASE     = -ldiffusion_maps
LDLIBS_DEBUG    = -lasan -lubsan
LDLIBS_TEST     = -lasan -lubsan
LDLIBS_TEST_PAR = -ltsan -lgomp
LDLIBS_RELEASE  = -lgomp

# ------------------------------------------------------------------------------

CPPFLAGS = $(CPPFLAGS_BASE) $(CPPFLAGS_$(PROFILE))
CXXFLAGS = $(CXXFLAGS_BASE) $(CXXFLAGS_$(PROFILE))
LDFLAGS  = $(LDFLAGS_BASE) $(LDFLAGS_$(PROFILE))
LDLIBS   = $(LDLIBS_BASE) $(LDLIBS_$(PROFILE))

MOD       = $(MOD_NAME)$(shell python3-config --extension-suffix)
BUILD_DIR = ../build/$(PROFILE)
LIB_DIR   = ../build/$(PROFILE)

.PHONY: all

all: $(BUILD_DIR)/$(MOD)

$(BUILD_DIR)/$(MOD): $(BUILD_DIR)/mod.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -shared -o $@

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -fPIC -MMD -MF $(BUILD_DIR)/$*.d -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

-include $(BUILD_DIR)/*.d
