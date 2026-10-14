# docs/Makefile

SPHINXBUILD   = sphinx-build
SPHINXAPIDOC  = sphinx-apidoc
SPHINXOPTS    ?=
SOURCEDIR     = .
BUILDDIR      = _build
APIDIR        = api
PKGDIR        = ../src/jnkepler

.PHONY: apidoc html clean

apidoc:
	$(SPHINXAPIDOC) -f -o $(APIDIR) $(PKGDIR)

html:
	$(SPHINXBUILD) $(SPHINXOPTS) -b html $(SOURCEDIR) $(BUILDDIR)/html

clean:
	rm -rf $(BUILDDIR)
