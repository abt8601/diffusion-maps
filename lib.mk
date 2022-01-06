CPPFLAGS_TEST_PAR = -DPAR
CPPFLAGS_RELEASE  = -DNDEBUG -DPAR

CXXFLAGS_BASE     = -std=c++17 -pedantic -Wall -Wextra -Werror -Iinclude
CXXFLAGS_DEBUG    = -g -fsanitize=address -fsanitize=undefined
CXXFLAGS_TEST     = -g -fsanitize=address -fsanitize=undefined -Og
CXXFLAGS_TEST_PAR = -g -fsanitize=thread -fopenmp -Og
CXXFLAGS_RELEASE  = -fopenmp -O3 -funroll-loops -march=native

# ------------------------------------------------------------------------------

CPPFLAGS = $(CPPFLAGS_BASE) $(CPPFLAGS_$(PROFILE))
CXXFLAGS = $(CXXFLAGS_BASE) $(CXXFLAGS_$(PROFILE))

BUILD_DIR = build/$(PROFILE)

.PHONY: all

all: $(BUILD_DIR)/diffusion_maps.a

$(BUILD_DIR)/diffusion_maps.a: $(BUILD_DIR)/diffusion_maps.o $(BUILD_DIR)/eig_solver.o
	$(AR) $(ARFLAGS) $@ $^

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -fPIC -MMD -MF $(BUILD_DIR)/$*.d -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

-include $(BUILD_DIR)/*.d
