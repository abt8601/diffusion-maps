# Flags

CXXFLAGS = -std=c++17 -pedantic -Wall -Wextra -Werror -I../include -g -fsanitize=address -fsanitize=undefined
LDLIBS   = -lasan -lubsan -lcriterion

# ------------------------------------------------------------------------------

BUILD_DIR = build
LIB = ../build/$(PROFILE)/diffusion_maps.a
TESTS := $(addprefix $(BUILD_DIR)/,$(basename $(wildcard test_*)))

.PHONY: test clean

test: $(TESTS)
	for test in $(TESTS); do echo "Running $$test"; $$test; done

$(BUILD_DIR)/test_%: $(BUILD_DIR)/test_%.o
	$(CXX) $(LDFLAGS) $^ $(LIB) $(LDLIBS) -o $@

$(BUILD_DIR)/test_%.o: test_%.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -fPIC -MMD -MF $(BUILD_DIR)/test_$*.d -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	$(RM) -r build

-include $(BUILD_DIR)/*.d
