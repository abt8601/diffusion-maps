CPPFLAGS_TEST_PAR = -DPAR

CXXFLAGS_BASE     = -std=c++17 -pedantic -Wall -Wextra -Werror -I../include
CXXFLAGS_TEST     = -g -fsanitize=address -fsanitize=undefined -Og
CXXFLAGS_TEST_PAR = -g -fsanitize=thread -fopenmp -Og

LDFLAGS_BASE = -L$(LIB_DIR)

LDLIBS_BASE     = -ldiffusion_maps
LDLIBS_TEST     = -lasan -lubsan -lcriterion
LDLIBS_TEST_PAR = -ltsan -lgomp -lcriterion

# ------------------------------------------------------------------------------

CPPFLAGS = $(CPPFLAGS_BASE) $(CPPFLAGS_$(PROFILE))
CXXFLAGS = $(CXXFLAGS_BASE) $(CXXFLAGS_$(PROFILE))
LDFLAGS  = $(LDFLAGS_BASE) $(LDFLAGS_$(PROFILE))
LDLIBS   = $(LDLIBS_BASE) $(LDLIBS_$(PROFILE))

BUILD_DIR  = ../build/tests/$(PROFILE)
LIB_DIR    = ../build/$(PROFILE)
TESTS     := $(addprefix $(BUILD_DIR)/,$(basename $(wildcard test_*.cpp)))

.PHONY: test

test: $(TESTS)
	for test in $(TESTS); do echo "Running $$(basename $$test)"; $(TEST_ENV) $$test; done

$(BUILD_DIR)/test_%: $(BUILD_DIR)/test_%.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/test_%.o: test_%.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -fPIC -MMD -MF $(BUILD_DIR)/test_$*.d -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

-include $(BUILD_DIR)/*.d
