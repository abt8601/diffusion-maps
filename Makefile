DEFAULT_PROFILE = DEBUG
PROFILE         = $(DEFAULT_PROFILE)

TSAN = $(shell ldconfig -p | awk '$$1 ~ /^libtsan/ { print $$1 }')
PAR_TEST_ENV = LD_PRELOAD=$(TSAN) OMP_NUM_THREADS=2

.PHONY: all lib pymod test cpptest pytest clean

all: lib pymod

lib:
	$(MAKE) -f lib.mk PROFILE=$(PROFILE)

pymod: lib
	$(MAKE) -C pybind PROFILE=$(PROFILE)
	cp build/$(PROFILE)/*.so .

test:
	$(MAKE) cpptest
	$(MAKE) pytest

cpptest:
	$(MAKE) lib PROFILE=TEST
	$(MAKE) -C tests PROFILE=TEST

	# The tests for the parallel version crashes Criterion, so they are disabled
	# for now.
	#$(MAKE) lib PROFILE=TEST_PAR
	#$(MAKE) -C tests PROFILE=TEST_PAR TEST_ENV="$(PAR_TEST_ENV)"

pytest: $(MOD)
	# Run the tests for the parallel version only, in order to save time.
	$(MAKE) pymod PROFILE=TEST_PAR
	$(PAR_TEST_ENV) python3 -m pytest

clean:
	$(RM) -r build
