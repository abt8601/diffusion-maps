DEFAULT_PROFILE = DEBUG
PROFILE         = $(DEFAULT_PROFILE)

TSAN = $(shell ldconfig -p | awk '$$1 ~ /^libtsan/ { print $$1 }')
PAR_TEST_ENV = OMP_NUM_THREADS=2

.PHONY: all lib pymod test cpptest pytest doc cppdoc cppdoc-html cppdoc-pdf \
	cppdoc-doxygen pydoc clean

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

pytest: $(MOD)
	$(MAKE) pymod PROFILE=RELEASE
	$(PAR_TEST_ENV) python3 -m pytest

doc: cppdoc pydoc

cppdoc: cppdoc-html cppdoc-pdf

cppdoc-html: cppdoc-doxygen

cppdoc-pdf: cppdoc-doxygen
	$(MAKE) -C build/cppdoc/latex

cppdoc-doxygen: | build
	doxygen Doxyfile

pydoc: | build
	pdoc --docformat numpy diffusion_maps -o build/pydoc

build:
	mkdir build

clean:
	$(RM) -r build
