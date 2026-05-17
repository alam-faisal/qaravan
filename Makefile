SPHINXBUILD = sphinx-build
SOURCEDIR   = docs/source
BUILDDIR    = docs/build

.PHONY: docs clean-docs

docs:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)

clean-docs:
	rm -rf $(BUILDDIR)
